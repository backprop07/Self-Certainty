import json 
from collections import defaultdict
import os 
from tabulate import tabulate 
import re
import sys 
from eval_utils import load_model_results, extract_values_from_json, extract_first_complete_json, model_specific_extraction, model_name_replacement
from livebench_math_utils.AMPS_Hard.utils import amps_hard_process_results
from livebench_math_utils.AMPS_Hard.utils import amps_hard_process_results_with_processed_output
from livebench_math_utils.math_competitions.utils import mathcontest_process_results, aime_process_results
from livebench_math_utils.olympiad.utils import proof_rearrangement_process_results
import argparse
from eval_utils_padding import is_amps_hard, sanitize_math_answers, convert_AAAAA_to_A


# def livebench_implemented_correct(data):

#     task_or_subtask = data["subtask"] if "subtask" in data.keys() else data["task"]
#     question_text = data["problem"]
#     ground_truth = data["answer"]
#     llm_answer = data["output"][0]
#     score = 0
#     splits = task_or_subtask.split('_')
#     if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):
#         score = mathcontest_process_results(ground_truth, llm_answer, question_text)
#     elif splits[0] == "aime":
#         score = aime_process_results(ground_truth, llm_answer)
#     elif splits[0] in ["imo", "usamo"]:
#         score = proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True)
#     elif "amps_hard" in task_or_subtask:
#         score = amps_hard_process_results(ground_truth, llm_answer)
#     return score==1

# def is_MCQA(data):
#     mc = False
#     task_or_subtask = data["subtask"] if "subtask" in data.keys() else data["task"]
#     splits = task_or_subtask.split('_')
#     if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):
#         mc = True
#     return mc

# def is_aime(data):
#     task_or_subtask = data["subtask"] if "subtask" in data.keys() else data["task"]
#     splits = task_or_subtask.split('_')
#     if splits[0] == "aime":
#         return True
#     return False
        
# def is_imo(data):
#     task_or_subtask = data["subtask"] if "subtask" in data.keys() else data["task"]
#     splits = task_or_subtask.split('_')
#     if splits[0] in ["imo", "usamo"]:
#         return True
#     return False

def eval_model(model, filepath):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_answer = 0  
    
    reason_lens = []
    parsed_results = [] 
    for item in data:  
        model_dir = item["generator"]
        # Read and Parse the prediction from model output
        parsed_item = item.copy()
        prediction_str = item["output"][min(0, len(item["output"])-1)]
        prediction_json = extract_first_complete_json(prediction_str)
        flag_parsed_answer = True
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            # print("-")
        if prediction_json is None or "answer" not in prediction_json: 
            try_extracted_answer = model_specific_extraction(model_dir, prediction_str)
            if try_extracted_answer:
                # print(f"Extracted answer from model: {try_extracted_answer}")
                prediction_json["answer"] = try_extracted_answer
            else:
                no_answer += 1 
                flag_parsed_answer = False 
       
        reason = prediction_json.get("reasoning", "")
        correct_answer = item["answer"].replace("#", "").strip()
        model_answer = None 
        if not flag_parsed_answer:
            # if "{"+correct_answer+"}" in prediction_str:
            # # Note: we assume the answer is correct if it is in the prediction string
            #     parsed_item["remarks"] = "Correct answer + {} is in the prediction string"
            #     model_answer = correct_answer
            # else:
            parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None} # not matched 
            parsed_item["correct_answer"] = {"raw": correct_answer}
            parsed_item["matched"] = "No answer extracted"
            parsed_results.append(parsed_item) 
            continue
        else:
            model_answer = str(prediction_json["answer"])
            
        # sanitize the answers
        model_answer = convert_AAAAA_to_A(model_answer)
        
        raw_model_answer = model_answer[:]
        model_answer = sanitize_math_answers(model_answer)
        correct_answer = sanitize_math_answers(correct_answer)
        
        first_number_in_model_answer = re.search(r"-?\d+(\.\d+)?", model_answer)
        first_number_in_correct_answer = re.search(r"-?\d+(\.\d+)?", correct_answer)
        correct = False 
        # if is_MCQA(item):
        #     if model_answer == correct_answer:
        #         correct = True
        # elif livebench_implemented_correct(item):
        #     correct = True
        # # elif first_number_in_model_answer and first_number_in_correct_answer:
        # #     if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
        # #         correct = True
        # # elif model_answer == correct_answer:
        # #     correct = True
        
        #only amps hard and MCQA differs; imo one response repeat answer 3 times, 1,1,1, can be treated as correct or incorrect, consider it incorrect.
        if model_answer == correct_answer:
            correct = True
        elif "livebench" not in item["dataset"]:
            if first_number_in_correct_answer and first_number_in_model_answer:
                if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
                    correct = True
        elif is_amps_hard(item) and amps_hard_process_results_with_processed_output(correct_answer, model_answer):
            correct = True

        
        if correct:
            solved_examples += 1

        reason_lens.append(len(reason))

        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = {"raw": raw_model_answer, "sanitized": model_answer, "first_number": first_number_in_model_answer.group() if first_number_in_model_answer else None}
        parsed_item["correct_answer"] = {"raw": correct_answer, "sanitized": correct_answer, "first_number": first_number_in_correct_answer.group() if first_number_in_correct_answer else None}
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)


 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results

def eval_model_best(model, filepath, best_N):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_answer = 0  
    
    reason_lens = []
    parsed_results = [] 
    for item in data:
        model_dir = item["generator"]
        find_response = False
        if best_N == -1:
            best_N = len(item["output"])
        best_N = min(best_N, len(item["output"]))
        for i in range(best_N):
            # Read and Parse the prediction from model output
            parsed_item = item.copy()
            prediction_str = item["output"][i] 
            prediction_json = extract_first_complete_json(prediction_str)
            flag_parsed_answer = True
            if prediction_json is None or "answer" not in prediction_json:
                prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
                # print("-")
            if prediction_json is None or "answer" not in prediction_json: 
                try_extracted_answer = model_specific_extraction(model_dir, prediction_str)
                if try_extracted_answer:
                    # print(f"Extracted answer from model: {try_extracted_answer}")
                    prediction_json["answer"] = try_extracted_answer
                else:
                    continue 
            find_response = True      
            reason = prediction_json.get("reasoning", "")
            correct_answer = item["answer"].replace("#", "").strip()
            model_answer = None 
            if not flag_parsed_answer:
                # if "{"+correct_answer+"}" in prediction_str:
                # # Note: we assume the answer is correct if it is in the prediction string
                #     parsed_item["remarks"] = "Correct answer + {} is in the prediction string"
                #     model_answer = correct_answer
                # else:
                parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None} # not matched 
                parsed_item["correct_answer"] = {"raw": correct_answer}
                parsed_item["matched"] = "No answer extracted"
                parsed_results.append(parsed_item) 
                continue
            else:
                model_answer = str(prediction_json["answer"])
            model_answer = convert_AAAAA_to_A(model_answer)
            # sanitize the answers
            raw_model_answer = model_answer[:]
            model_answer = sanitize_math_answers(model_answer)
            correct_answer = sanitize_math_answers(correct_answer)
            
            first_number_in_model_answer = re.search(r"-?\d+(\.\d+)?", model_answer)
            first_number_in_correct_answer = re.search(r"-?\d+(\.\d+)?", correct_answer)
            correct = False 
            if model_answer == correct_answer:
                correct = True
            elif "livebench" not in item["dataset"]:
                if first_number_in_correct_answer and first_number_in_model_answer:
                    if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
                        correct = True
            elif is_amps_hard(item) and amps_hard_process_results_with_processed_output(correct_answer, model_answer):
                correct = True
                    
            if correct:
                solved_examples += 1
                break

        if not find_response:
            no_answer += 1
            parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None}
            
        reason_lens.append(len(reason))
        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = {"raw": raw_model_answer, "sanitized": model_answer, "first_number": first_number_in_model_answer.group() if first_number_in_model_answer else None}
        parsed_item["correct_answer"] = {"raw": correct_answer, "sanitized": correct_answer, "first_number": first_number_in_correct_answer.group() if first_number_in_correct_answer else None}
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)

 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results

def eval_model_first_answered(model, filepath, best_N):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0 
    num_total_examples = len(data) 
    no_answer = 0  
    
    reason_lens = []
    parsed_results = [] 
    for item in data:  
        model_dir = item["generator"]
        # Read and Parse the prediction from model output
        parsed_item = item.copy()
        if best_N == -1:
            best_N = len(item["output"])
        best_N = min(best_N, len(item["output"]))
        for i in range(best_N):
            flag_parsed_answer = True
            prediction_str = item["output"][i]
            prediction_json = extract_first_complete_json(prediction_str)
            if prediction_json is None or "answer" not in prediction_json:
                prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            if prediction_json is None or "answer" not in prediction_json: 
                try_extracted_answer = model_specific_extraction(model_dir, prediction_str)
                if try_extracted_answer:
                    prediction_json["answer"] = try_extracted_answer
                else:
                    flag_parsed_answer = False
                    continue
            if flag_parsed_answer:
                break 
        if not flag_parsed_answer:
            no_answer += 1 
            parsed_item["model_answer"] = {"raw": None, "sanitized": None, "first_number": None}    
            parsed_item["correct_answer"] = {"raw": item["answer"].replace("#", "").strip()}
            parsed_item["matched"] = "No answer extracted"
            parsed_results.append(parsed_item)
            continue
        reason = prediction_json.get("reasoning", "")
        correct_answer = item["answer"].replace("#", "").strip()
        model_answer = str(prediction_json["answer"])
        model_answer = convert_AAAAA_to_A(model_answer)
        raw_model_answer = model_answer[:]
        model_answer = sanitize_math_answers(model_answer)
        correct_answer = sanitize_math_answers(correct_answer)
        
        first_number_in_model_answer = re.search(r"-?\d+(\.\d+)?", model_answer)
        first_number_in_correct_answer = re.search(r"-?\d+(\.\d+)?", correct_answer)
        correct = False 
        if model_answer == correct_answer:
            correct = True
        elif "livebench" not in item["dataset"]:
            if first_number_in_correct_answer and first_number_in_model_answer:
                if float(first_number_in_model_answer.group()) == float(first_number_in_correct_answer.group()):
                    correct = True
        elif is_amps_hard(item) and amps_hard_process_results_with_processed_output(correct_answer, model_answer):
            correct = True
        if correct:
            solved_examples += 1
            
        reason_lens.append(len(reason))
        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = {"raw": raw_model_answer, "sanitized": model_answer, "first_number": first_number_in_model_answer.group() if first_number_in_model_answer else None}
        parsed_item["correct_answer"] = {"raw": correct_answer, "sanitized": correct_answer, "first_number": first_number_in_correct_answer.group() if first_number_in_correct_answer else None}
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)

 
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"
    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results

def gen_results(run_name_folders, mode="eval", best_N=-1):
    model_results = load_model_results(run_name_folders)

    columns = ["Model", "Mode", "Acc", "No answer", "Total", "Reason Lens"]
    rows = []
    for model_name, filepath in model_results.items(): 
        # print(model_name)
        # if model_name in ["gemini-1.5-flash-exp-0827%greedy"]:
        #     continue
        if mode == "best":
            result, parsed_results = eval_model_best(model_name, filepath, best_N)
        elif mode == "first_answered":
            result, parsed_results = eval_model_first_answered(model_name, filepath, best_N)
        else:
            result, parsed_results = eval_model(model_name, filepath) 
        # save the parsed_results to the same filepath with a  new prefix 
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        # create folders if not exist
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        # save 
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)
        rows.append(result)

    # sort the rows by puzzle accuracy
    rows = sorted(rows, key=lambda x: -float(x["Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))
    # print(tabulate(rows, headers=columns, tablefmt="github"))

    # write to markdown file
    banner_header = """
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/4666e72d-4202-4283-8e78-e5ce2b030dcf" alt="zebra_banner" style="width: 69%;" />
</div>


"""
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(banner_header+tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center"))

    # write to json file 
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on livebench math dataset")
    parser.add_argument("--dataset", type=str, default="livebench-math", help="Dataset name")
    parser.add_argument("--mode", type=str, default="eval", help="Mode of the evaluation")
    parser.add_argument("--best_N", type=int, default=-1, help="Number of best results to consider")
    args = parser.parse_args()

    data_name = args.dataset
    mode = args.mode
 
    run_name_folders = {
        mode: f"result_dirs/{data_name}", 
        # "sampling": f"result_dirs/{data_name}/sampling",
        # "greedy@no_cot": f"result_dirs/{data_name}/greedy@no_cot",
    } 
    if mode == "eval":
        gen_results(run_name_folders)
    elif mode == "best":
        gen_results(run_name_folders, mode="best", best_N=args.best_N)
    elif mode == "first_answered":
        gen_results(run_name_folders, mode="first_answered", best_N=args.best_N)
