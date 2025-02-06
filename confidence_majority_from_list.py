import argparse
import json
from evaluation.livebench_math_utils.AMPS_Hard.utils import two_answers_are_equiv
from evaluation.eval_utils import extract_values_from_json, extract_first_complete_json, model_specific_extraction
from evaluation.eval_utils_padding import is_amps_hard, sanitize_math_answers, convert_AAAAA_to_A, extract_answer_from_output
import torch
import tqdm
import os

# Set the environment variable for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def normalize_confidence(confidence: float) -> float:
    confidence = max(10, confidence)
    confidence = min(25, confidence)
    confidence = (confidence - 10) / 15
    return confidence

@torch.no_grad()    
def confidence_with_file(filepath, best_N=5, window_size=10, model="default", mode="middle", power=0.5):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Make item has attribute confidence list, by exmaine data[0]
    if "confidence_list" not in data[0]:
        raise ValueError("The input file does not have confidence_list attribute")
    if "generator" not in data[0]:
        raise ValueError("The input file does not have generator attribute")
    
    model_dir = data[0]["generator"]
    if data[0]["dataset"] == "crux":
        dataset = "crux"
    else:
        dataset = "math"
    print("Loaded")
    to_write = []
    
    best_N = min(best_N, len(data[0]["output"]))
    for index in tqdm.tqdm(range(len(data)), desc="Processing inputs"):
        item = data[index]
        new_item = {k: v for k, v in item.items() if k != "output"}
        all_confidences = item["confidence_list"]
        all_confidences = all_confidences[:best_N]
        outputs = item["output"][:best_N]

        # Perform Borda count based on confidences
        sorted_indices = sorted(range(len(all_confidences)), key=lambda k: all_confidences[k], reverse=True)
        votes_per_output = [len(all_confidences) - rank for rank in range(len(all_confidences))]
        
        # Exponetial votes
        # votes_per_output = [1.2**vote for vote in votes_per_output]        
        
        # Power function votes
        votes_per_output = [vote**power for vote in votes_per_output]
        
        
        votes_map = {sorted_indices[i]: votes_per_output[i] for i in range(len(sorted_indices))}
        votes = [0 for _ in range(len(all_confidences))]
        for i in range(len(all_confidences)):
            answer_i = extract_answer_from_output(outputs[i], model_dir, dataset)
            if answer_i is None:
                continue
            find_answer = False
            for j in range(i):
                answer_j = extract_answer_from_output(outputs[j], model_dir, dataset)
                if answer_j is None:
                    continue
                if answer_i == answer_j:
                    votes[j] += votes_map[i]
                    find_answer = True
                    break
                elif is_amps_hard(item):
                    if two_answers_are_equiv(answer_i, answer_j):
                        votes[j] += votes_map[i]
                        find_answer = True
                        break
            if not find_answer:
                votes[i] += votes_map[i]
                
        all_confidences = [votes[i] for i in range(len(all_confidences))]
        
        best_confidence = max(all_confidences)
        best_index = all_confidences.index(best_confidence)
        new_item["output"] = [outputs[best_index]]
        new_item["confidence"] = best_confidence
        to_write.append(new_item)
        # print(f"Processed {index+1}/{len(data)}")
    
    output_file = input_file.replace(".json", f"-confidence-borda-power-{power}-{best_N}-kl.json")
    print(f"Writing to {output_file}")
    with open(output_file, "w") as f:
        json.dump(to_write, f, indent=4)
    
# python3 src/confidence_majority_from_list.py --input_file /data/xuandong_zhao/mnt/zheweikang/ZeroEval/result_dirs/gsm/Llama3.1-samples-gsm-confidence-list.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--best_N", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=-1)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--power", type=float, default=0)
    args = parser.parse_args()
    input_file = args.input_file
    # print(f"Processing {input_file}")
    # self_consistency_file(input_file, best_N=args.best_N)

    # Call the function
    confidence_with_file(input_file, best_N=args.best_N, window_size=args.window_size, model=args.model, mode=args.mode, power=args.power)
            
        
    
        
    
    