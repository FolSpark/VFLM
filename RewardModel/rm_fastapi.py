from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import base64
import math
from PIL import Image
from datetime import datetime
from transformers.utils import cached_file
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from trl import AutoModelForCausalLMWithValueHead
from qwen_vl_utils import process_vision_info


def load_valuehead_params(path_or_repo_id: str) -> dict[str, torch.Tensor]:
    r"""Load value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": None, "token": None}
    err_text = ""

    try:
        from safetensors import safe_open

        vhead_file = cached_file(filename="value_head.safetensors", **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    try:
        vhead_file = cached_file(filename="value_head.bin", **kwargs)
        return torch.load(vhead_file, map_location="cpu")
    except Exception as err:
        err_text = str(err)

    print(f"Provided path ({path_or_repo_id}) does not contain value head weights: {err_text}.")
    print("Ignore the above message if you are not resuming the training of a value head model.")
    return None

def make_message(bg_img_b64_str, image_b64_str, text_content):
    system_prompt = (
        "You are an experienced layout designer, skilled at elegantly arranging the specified text on the background image provided by the user.\n"
        "You know well how to use unique aesthetic principles to design a professional and appealing layout. Please design the final layout plan according to the background image and text content provided by the user, and directly provide the typeset image to the user.\n"
    )
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "background-image.png:"},
                {"type": "image", "image": bg_img_b64_str, "max_pixels": 512 * 28 * 28},
                {"type": "text", "text": f"\nPlease arrange the following text on the background image: \n{text_content}"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_b64_str, "max_pixels": 512 * 28 * 28}
            ]
        }
    ]
    return messages

MEAN = -8.234379460548926
STD = 10.824762709743482
def norm(x):
    """Normalize the input value."""
    return (x - MEAN) / STD

# 模型和tokenizer路径
MODEL_PATH = "/Path/To/Your/Model"  # 替换为你的模型路径或Hugging Face Hub上的模型ID

# 创建FastAPI应用
app = FastAPI(title="Qwen2.5-VL-RM服务", description="基于Qwen2.5-VL和trl库的AI服务")

# 定义请求和响应模型
class InputData(BaseModel):
    prompt: str
    image1: str  # base64编码的图片1
    image2: str  # base64编码的图片2

class OutputData(BaseModel):
    score: float

# 全局模型和tokenizer
model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    try:
        # 加载模型和tokenizer
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="auto",
        )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        vhead_params = load_valuehead_params(MODEL_PATH)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
        model.eval()  # 设置模型为评估模式
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise HTTPException(status_code=500, detail="模型加载失败")

@app.post("/compute_score", response_model=OutputData)
async def compute_score(input: InputData):
    # 打印当前日期
    print(f"当前日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # 准备输入
        messages = make_message(
            bg_img_b64_str=input.image1,  # 假设image1是背景图
            image_b64_str=input.image2,   # 假设image2是需要排版的图片
            text_content=input.prompt    # 使用prompt作为文本内容
        )
        texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.pretrained_model.device)
        
        # 生成回答
        with torch.no_grad():
            output = model(**inputs)
            _, _, values = output
            score = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))
            # score = torch.sigmoid(score)  # 应用sigmoid函数将分数归一化到0-1范围
            score = score.item()  # 获取单个分数值
            score = norm(score)

        # 输出日志
        print(f"Prompt: {input.prompt}")
        print(f"Score: {score}")

        return {"score": score}
    except Exception as e:
        print(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail="生成失败")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
