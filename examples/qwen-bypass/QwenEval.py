'''
CTC Evaluation script for Qwen3VL-8B-Instruct on MMStar.
Original paper claims: 70.9%.
Our faster replication: 67.40%.
Configuration: No LLM Judge, force multiple choice, min/max pixel resize.
'''

import torch
from MMStarDataset import MMStar
from tqdm import tqdm
import json
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class CONFIG:
    MMSTAR_TSV_PATH = "MMStar.tsv"
    QWEN_PATH = "/path/to/your/Qwen3VL-8B-Instruct" 
    MODE = "VANILLA" # VANILLA or TEST-CTC. TEST-CTC is to show that using the extraction+putback API is exactly the same as using Qwen3VL in transformers.
    
    MAX_NEW_TOKENS = 512 # we force single-letter answer, so 512 is in reallity never reached.
    MIN_PIXELS = 768*28*28
    MAX_PIXELS = 1536*28*28 # 5120*28*28 may trigger OOM. Under 1536*28*28, max GPU memory < 20 GB.
    
    DO_SAMPLE = False # greedy output, no randomness
    
    DEVICE = "cuda:4"
    
    PROMPT_SUFFIX = "Please select the correct answer from the options above.\nYou MUST reply with one single letter (A/B/C/D) with no explanation."
    TARGET_FOLDER = "MMStar-CTC"
    

def infer(pred_json: str = "predictions.json",
          config_save_json: str = "config_used.json"):
    '''
    Run inference. Dump to json as inference goes.
    '''
    config_dict = {
        "MMSTAR_TSV_PATH": CONFIG.MMSTAR_TSV_PATH,
        "QWEN_PATH": CONFIG.QWEN_PATH,
        "MODE": CONFIG.MODE,
        "MAX_NEW_TOKENS": CONFIG.MAX_NEW_TOKENS,
        "MIN_PIXELS": CONFIG.MIN_PIXELS,
        "MAX_PIXELS": CONFIG.MAX_PIXELS,
        "DO_SAMPLE": CONFIG.DO_SAMPLE,
        "PROMPT_SUFFIX": CONFIG.PROMPT_SUFFIX,
    }
    # dump config to json
    os.makedirs(CONFIG.TARGET_FOLDER, exist_ok=True)
    with open(os.path.join(CONFIG.TARGET_FOLDER, config_save_json), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Evaluation Config saved to {os.path.join(CONFIG.TARGET_FOLDER, config_save_json)}")
    
    # make dataset
    dataset = MMStar(CONFIG.MMSTAR_TSV_PATH, prompt_suffix=CONFIG.PROMPT_SUFFIX)
    
    if CONFIG.MODE == "VANILLA":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            CONFIG.QWEN_PATH,
            device_map=CONFIG.DEVICE,
            attn_implementation="eager",
            dtype="bfloat16"
        )
        model.eval()
        
        processor = AutoProcessor.from_pretrained(CONFIG.QWEN_PATH,
                                                trust_remote_code=True,
                                                max_pixels=CONFIG.MAX_PIXELS,
                                                min_pixels=CONFIG.MIN_PIXELS,)
        
        os.makedirs(CONFIG.TARGET_FOLDER, exist_ok=True)
        save_path = os.path.join(CONFIG.TARGET_FOLDER, pred_json)
        
        results = []
        peak_gpu_GB_usage = 0

        for idx in tqdm(range(len(dataset)), desc="Inferencing"):
            item = dataset[idx]
            prompt = item["prompt"]
            image = item["image"]
            
            messages = [
                    {
                        "role": "user",
                        "content": 
                        [
                            {"type": "image",
                            "image": image,
                            },
                            {"type": "text", "text": prompt}
                        ],
                    }
                ]
            
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(CONFIG.DEVICE)
            
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=CONFIG.MAX_NEW_TOKENS,
                    do_sample=CONFIG.DO_SAMPLE,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                
            # check GPU memory usage
            gpu_GB_usage = torch.cuda.max_memory_allocated(device=CONFIG.DEVICE) / (1024 ** 3)
            peak_gpu_GB_usage = max(peak_gpu_GB_usage, gpu_GB_usage)
            torch.cuda.empty_cache()
            
            generated_ids = outputs[0][len(inputs["input_ids"][0]):]
            prediction = processor.decode(generated_ids, skip_special_tokens=True,clean_up_tokenization_spaces=False)
            
            result_item = {
                "index": item["index"],
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "category": item["category"],
                "l2_category": item["l2_category"],
                "bench": item["bench"],
                "prediction": prediction,
            }
            results.append(result_item)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
    elif CONFIG.MODE == "TEST-CTC":
        # API for CoFAI.
        # This branch is to prove that the API replicates exactly the behavior of the original Qwen3VL in transformers.
        from QwenVL_CTC import Qwen3VLWrapper
        wrapper = Qwen3VLWrapper(model_path=CONFIG.QWEN_PATH, 
                                 device=CONFIG.DEVICE,
                                 min_pixels=CONFIG.MIN_PIXELS,
                                 max_pixels=CONFIG.MAX_PIXELS,
                                 dtype="bfloat16")
        

        os.makedirs(CONFIG.TARGET_FOLDER, exist_ok=True)
        save_path = os.path.join(CONFIG.TARGET_FOLDER, pred_json)
        
        results = []
        peak_gpu_GB_usage = 0

        for idx in tqdm(range(len(dataset)), desc="Inferencing"):
            item = dataset[idx]
            prompt = item["prompt"]
            image = item["image"]
            
            # FEATURE EXTRACTION:
            pixel_values, image_grid_thw = wrapper.prepare_image(image)
            hidden_states = wrapper.extract_features(pixel_values, image_grid_thw) 
            
            # ==================================================== #
            # NOTE to CoFAI researchers:
            # The hidden_states to apply compression/decompression is here. 
            # its size is (1152,H/16,W/16), where H,W is raw image size, unless altered by min/max pixels.
            # ====================================================#
            
            # CONTINUE GENERATION:
            output_text = wrapper.generate_from_features(
                hidden_states,
                prompt=prompt,
                max_new_tokens=CONFIG.MAX_NEW_TOKENS,
                do_sample=CONFIG.DO_SAMPLE,
            )
            
            gpu_GB_usage = torch.cuda.max_memory_allocated(device=CONFIG.DEVICE) / (1024 ** 3)
            peak_gpu_GB_usage = max(peak_gpu_GB_usage, gpu_GB_usage)
            torch.cuda.empty_cache()
            
            result_item = {
                "index": item["index"],
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "category": item["category"],
                "l2_category": item["l2_category"],
                "bench": item["bench"],
                "prediction": output_text,
            }
            results.append(result_item)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved all predictions to {save_path}")
    print(f"Peak GPU memory usage: {peak_gpu_GB_usage:.2f} GB")

def eval(pred_json: str = "predictions.json", results_txt: str = "results.txt"):
    '''
    Run evaluation on inference.
    Uses 2 patterns: (1) last letter; (2) "Option letter + option text" pattern.
    '''
    load_path = os.path.join(CONFIG.TARGET_FOLDER, pred_json)
    with open(load_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    total = len(results)
    correct = 0
    category_correct = {}   # category -> [correct, total]
    l2_category_correct = {}  # l2_category -> [correct, total]
    
    for item in results:
        answer = item["answer"].strip()
        prediction = item["prediction"].strip("")
        
        # pattern 1: Last letter is prediction
        pred_letter = prediction[-1]
        is_correct = (pred_letter == answer)
        
        if not is_correct:
            # pattern 2: The entire output is under the format "A. (option A text)"
            options = item["options"]
            for (letter,optiontext) in options.items():
                if prediction.strip(".") == letter+". "+optiontext.strip("."):
                    pred_letter = letter
                    is_correct = (pred_letter == answer)
                    break
                
        
        if is_correct:
            correct += 1
        
        # statistics on categories and level-2 categories
        cat = item["category"]
        if cat is not None:
            if cat not in category_correct:
                category_correct[cat] = [0, 0]
            category_correct[cat][1] += 1
            if is_correct:
                category_correct[cat][0] += 1
        
        l2 = item["l2_category"]
        if l2 is not None:
            if l2 not in l2_category_correct:
                l2_category_correct[l2] = [0, 0]
            l2_category_correct[l2][1] += 1
            if is_correct:
                l2_category_correct[l2][0] += 1
    
    # Write report
    total_acc = correct / total * 100 if total > 0 else 0
    
    lines = []
    lines.append(f"Total accuracy: {correct}/{total} = {total_acc:.2f}%")
    lines.append("\nCategory accuracies:")
    for cat, (c, t) in sorted(category_correct.items()):
        acc = c / t * 100 if t > 0 else 0
        lines.append(f"  {cat}: {c}/{t} = {acc:.2f}%")
    lines.append("\nL2 Category accuracies:")
    for l2, (c, t) in sorted(l2_category_correct.items()):
        acc = c / t * 100 if t > 0 else 0
        lines.append(f"  {l2}: {c}/{t} = {acc:.2f}%")
    
    report = "\n".join(lines)
    print(report)
    
    # save report
    save_path = os.path.join(CONFIG.TARGET_FOLDER, results_txt)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Results saved to {save_path}")
    
    return report


if __name__ == "__main__":
    infer()
    eval()