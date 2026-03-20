import os
import re
import json
import megfile
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
from verl.utils.svg_utils import export_svg_with_bg, export_svg_to_img


def extract_answer(action_string: str):
    answer = re.search(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
    return answer.group(1)
    
def extract_action(action_string: str):
    """
    Extracts the tool call from the action string.
    
    Args:
        action_string: The string containing the tool call in XML tags.
        
    Returns:
        A dictionary with the tool name and arguments.
        
    Raises:
        ValueError: If no tool call is found or JSON is invalid.
    """
    # Find all tool_call matches
    tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
    if not tool_call_matches:
        return []
    
    tool_calls = []
    for tool_call in tool_call_matches:
        try:
            tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool call: {e}")
    
    return tool_calls


def extract_tool_and_svg(text: str):
    # 提取工具名
    tool_start = text.find("TOOL:")
    if tool_start == -1:
        return None, None
    
    tool_name_start = tool_start + len("TOOL:")
    tool_name_end = text.find("\n", tool_name_start)
    if tool_name_end == -1:
        tool_name_end = len(text)
    
    tool_name = text[tool_name_start:tool_name_end].strip()
    
    params_start = text.find("PARAMS:", tool_name_end)
    if params_start == -1:
        return tool_name, None
    params_start += len("PARAMS:")
    params_string = text[params_start:].strip()
    if params_string.startswith("svg_code:"):
        params_string = params_string[len("svg_code:"):].strip()
    return tool_name, params_string


def process_entry(entry_data):
    index, data, output_dir, test_dataset = entry_data
    index_str = str(index)
    output_dir = os.path.join(output_dir, str(index).zfill(3))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        image_path = os.path.join(output_dir, f"{str(index).zfill(3)}-answer.png")
        if os.path.exists(image_path):
            return
        bg_image = Image.open(os.path.join("datasets/svg-data/data/", test_dataset[index_str]['image']))
        bg_image.save(os.path.join(output_dir, f"{str(index).zfill(3)}-bg.png"))
        
        output = data['output']
        
        turn = 0
        turns_output: str = output
        while '<tool_response>' in turns_output:
            turns = turns_output.split('\nuser\n<tool_response>', maxsplit=1)
            with open(os.path.join(output_dir, f"{str(index).zfill(3)}-{turn}.txt"), 'w') as f:
                f.write(turns[0])
            turns_output = turns[1]
            turn += 1
        with open(os.path.join(output_dir, f"{str(index).zfill(3)}-{turn}.txt"), 'w') as f:
            f.write(turns_output)
        
        # Extract the answer
        try:
            answer = extract_answer(output)
            if answer.startswith("```svg\n") and answer.endswith("```"):
                svg_code = answer[7:-3].strip()
            else:
                svg_code = answer
            image_answer = export_svg_to_img(svg_code, bg_image)
            # image_answer = export_svg_with_bg(svg_code, bg_image)
            # Save the image
            image_answer.save(image_path)
        except Exception as e:
            print(f"Error extracting answer for entry {index}: {str(e)}")
            # return
        
        for idx, tool_call in enumerate(extract_action(output)):
            tool_name, svg_code = extract_tool_and_svg(tool_call)
            if tool_name == "svg_to_image_tool":
                if svg_code.startswith("```svg\n") and svg_code.endswith("```"):
                    svg_code = svg_code[7:-3].strip()
                image_tool = export_svg_to_img(svg_code, bg_image)
                # image_tool = export_svg_with_bg(svg_code, bg_image)
                # Save the image
                image_path = os.path.join(output_dir, f"{str(index).zfill(3)}-{idx}.png")
                image_tool.save(image_path)
        
    except Exception as e:
        print(f"Error processing entry {index}: {str(e)}")


if __name__ == "__main__":
    rollout_dir = "work_dirs/tool_rl_grpo_svg-only_result-2025-09-05/val"
    test_dataset = json.load(open("infer/test_rl_w_tool.json", "r"))
    
    rollout_files = megfile.smart_glob(os.path.join(rollout_dir, "*.jsonl"))
    rollout_files.sort()
    
    # Number of processes to use
    num_processes = 64  # Adjust based on your system's capabilities
    
    for file in rollout_files:
        output_dir = os.path.join(rollout_dir, os.path.basename(file).split('.')[0])
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing file: {file}")
        
        # Read all entries from the JSONL file
        entries = []
        with megfile.smart_open(file, 'r') as f:
            for index, line in enumerate(f):
                data = json.loads(line.strip())
                entries.append((index, data, output_dir, test_dataset))
        
        # Process entries in parallel
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(process_entry, entries), total=len(entries), desc="Processing entries"))
        
        print(f"Completed processing {len(entries)} entries from {file}")
