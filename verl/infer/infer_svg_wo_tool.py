import os
import re
import json
import megfile
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
from verl.utils.svg_utils import export_svg_to_img


def extract_answer(action_string: str):
    answer = re.search(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
    return answer.group(1)



def process_entry(entry_data):
    index, data, output_dir, test_dataset = entry_data
    index_str = str(index)
    output_dir = os.path.join(output_dir, str(index).zfill(3))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        image_path = os.path.join(output_dir, f"{str(index).zfill(3)}-answer.png")
        if os.path.exists(image_path):
            return
        
        output = data['output']
        open(os.path.join(output_dir, f"{str(index).zfill(3)}-output.txt"), "w").write(output)
        # return
        
        bg_image = Image.open(os.path.join("datasets/svg-data/data/", test_dataset[index_str]['image']))
        bg_image.save(os.path.join(output_dir, f"{str(index).zfill(3)}-bg.png"))
        
        # Extract the answer
        try:
            answer = extract_answer(output)
            if answer.startswith("```svg\n") and answer.endswith("```"):
                svg_code = answer[7:-3].strip()
            else:
                svg_code = answer
            image_answer = export_svg_to_img(svg_code, bg_image)
            # Save the image
            image_answer.save(image_path)
        except Exception as e:
            print(f"Error extracting answer for entry {index}: {str(e)}")
            # return

        
    except Exception as e:
        print(f"Error processing entry {index}: {str(e)}")


if __name__ == "__main__":
    rollout_dir = "work_dirs/tool_rl_grpo_svg_thinking_7b_from_sft-2025-08-07-bs1024-lr1e-6/val"
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
