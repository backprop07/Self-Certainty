import argparse
import json
from evaluation.eval_utils_padding import extract_answer_from_output
import tqdm

 
def confidence_with_file(filepath, best_N=5):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Make item has attribute confidence list, by exmaine data[0]
    if "confidence_list" not in data[0]:
        raise ValueError("The input file does not have confidence_list attribute")
    model_dir = data[0]["generator"]
    to_write = []
    
    best_N = min(best_N, len(data[0]["output"]))
    for index in tqdm.tqdm(range(len(data)), desc="Processing inputs"):
        item = data[index]
        new_item = {k: v for k, v in item.items() if k != "output"}
        all_confidences = item["confidence_list"]
        all_confidences = all_confidences[:best_N]
        outputs = item["output"][:best_N]
        
        allow_filtering_not_completed_answers = True
        if allow_filtering_not_completed_answers:
            for j in range(len(outputs)):
                if extract_answer_from_output(outputs[j], model_dir) is None:
                    all_confidences[j] = -1e9
                    
    
        best_confidence = max(all_confidences)
        best_index = all_confidences.index(best_confidence)
        new_item["output"] = [outputs[best_index]]
        new_item["confidence"] = best_confidence
        to_write.append(new_item)
    
    output_file = input_file.replace(".json", f"-confidence-{best_N}.json")
    print(f"Writing to {output_file}")
    with open(output_file, "w") as f:
        json.dump(to_write, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--best_N", type=int, default=4)
    args = parser.parse_args()
    input_file = args.input_file
    confidence_with_file(input_file, best_N=args.best_N)
            
        
    
        
    
    