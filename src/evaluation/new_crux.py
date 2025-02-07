import json
import os
import re
import sys
import argparse
from collections import defaultdict

from tabulate import tabulate

# Make sure these utilities exist or adapt to your own helper methods
# If you don't have these, adjust or remove them as needed.
from eval_utils import (
    load_model_results,
    extract_values_from_json,
    extract_first_complete_json,
    model_specific_extraction,
    model_name_replacement
)

def eval_model(model: str, filepath: str):
    """
    Evaluate the model on a CRUX-like dataset. 
    Each item is assumed to have:
      - an "output" list of strings (model outputs)
      - an "answer" field for the correct answer

    Args:
        model: A string of the form "ModelName%Mode", e.g. "gpt4%greedy"
        filepath: Path to the JSON file containing predictions
    """
    print(f"Processing {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)

    solved_examples = 0
    num_total_examples = len(data)
    no_answer = 0

    reason_lens = []
    parsed_results = []

    for item in data:
        # By default, look at the first output
        prediction_str = item["output"][0] if item["output"] else ""
        prediction_json = extract_first_complete_json(prediction_str)

        # Attempt fallback if no JSON or missing "answer"
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
        if prediction_json is None or "answer" not in prediction_json:
            try_extracted_answer = model_specific_extraction(model, prediction_str)
            if not try_extracted_answer:
                no_answer += 1
                # Prepare partial record
                parsed_item = item.copy()
                parsed_item["model_answer"] = {"raw": None, "sanitized": None}
                parsed_item["matched"] = "No answer extracted"
                parsed_results.append(parsed_item)
                continue
            else:
                prediction_json = prediction_json or {}
                prediction_json["answer"] = try_extracted_answer

        # Retrieve reasoning if present
        reason = prediction_json.get("reasoning", "")

        # Sanitize
        model_answer = str(prediction_json["answer"])
        sanitized_model_answer = model_answer.strip("'\"").replace('\n', '\\n')
        correct_answer = str(item["answer"]).strip("'\"")

        # Simple string match for correctness
        correct = (sanitized_model_answer == correct_answer)

        if correct:
            solved_examples += 1

        reason_lens.append(len(reason))

        parsed_item = item.copy()
        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = {
            "raw": model_answer,
            "sanitized": sanitized_model_answer
            
        }
        parsed_item["correct_answer"] = {
            "raw": item["answer"],
            "sanitized": correct_answer
        }
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)

    acc = (solved_examples / num_total_examples * 100) if num_total_examples else 0
    no_ans_rate = (no_answer / num_total_examples * 100) if num_total_examples else 0
    avg_reason_len = (sum(reason_lens) / len(reason_lens)) if reason_lens else 0

    result = {
        "Model": model_name_replacement(model.split("%")[0]),
        "Mode": "eval",
        "Acc": f"{acc:.2f}",
        "No answer": f"{no_ans_rate:.2f}",
        "Total": num_total_examples,
        "Reason Lens": f"{avg_reason_len:.2f}",
    }
    return result, parsed_results


def eval_model_best(model: str, filepath: str, best_N: int):
    """
    Evaluate with up to best_N attempts per item, stopping if any attempt is correct.
    If best_N = -1, use all attempts.
    """
    print(f"Processing {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)

    solved_examples = 0
    num_total_examples = len(data)
    no_answer = 0

    reason_lens = []
    parsed_results = []

    for item in data:
        outputs = item["output"]
        best_N_local = len(outputs) if best_N == -1 else best_N
        best_N_local = min(best_N_local, len(outputs))

        found_any_answer = False
        found_correct = False
        reason_str = ""

        for i in range(best_N_local):
            prediction_str = outputs[i]
            prediction_json = extract_first_complete_json(prediction_str)

            if prediction_json is None or "answer" not in prediction_json:
                prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            if prediction_json is None or "answer" not in prediction_json:
                try_extracted_answer = model_specific_extraction(model, prediction_str)
                if not try_extracted_answer:
                    continue
                else:
                    prediction_json = prediction_json or {}
                    prediction_json["answer"] = try_extracted_answer

            found_any_answer = True

            # Get reasoning to keep track of length (could be from the last attempt)
            reason_str = prediction_json.get("reasoning", "")

            # Sanitize
            model_answer = str(prediction_json["answer"])
            sanitized_model_answer = model_answer.strip("'\"").replace('\n', '\\n')
            correct_answer = str(item["answer"]).strip("'\"")

            # Simple correctness check
            if sanitized_model_answer  == correct_answer:
                solved_examples += 1
                found_correct = True
                break  # Stop checking further attempts

        if not found_any_answer:
            no_answer += 1

        reason_lens.append(len(reason_str))

        # Save record for analysis
        parsed_item = item.copy()
        parsed_item["reasoning"] = reason_str
        parsed_item["model_answer"] = {
            "raw": model_answer if found_any_answer else None,
            "sanitized": sanitized_model_answer if found_any_answer else None
        }
        parsed_item["correct_answer"] = {
            "raw": item["answer"],
            "sanitized": correct_answer
        }
        parsed_item["matched"] = found_correct
        parsed_results.append(parsed_item)

    acc = (solved_examples / num_total_examples * 100) if num_total_examples else 0
    no_ans_rate = (no_answer / num_total_examples * 100) if num_total_examples else 0
    avg_reason_len = (sum(reason_lens) / len(reason_lens)) if reason_lens else 0

    result = {
        "Model": model_name_replacement(model.split("%")[0]),
        "Mode": "best",
        "Acc": f"{acc:.2f}",
        "No answer": f"{no_ans_rate:.2f}",
        "Total": num_total_examples,
        "Reason Lens": f"{avg_reason_len:.2f}",
    }
    return result, parsed_results


def eval_model_first_answered(model: str, filepath: str, best_N: int):
    """
    Evaluate the very first output that contains a valid answer (JSON or extracted).
    If none of the outputs (up to best_N) has an answer, it counts as no answer.
    """
    print(f"Processing {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)

    solved_examples = 0
    num_total_examples = len(data)
    no_answer = 0

    reason_lens = []
    parsed_results = []

    for item in data:
        outputs = item["output"]
        best_N_local = len(outputs) if best_N == -1 else best_N
        best_N_local = min(best_N_local, len(outputs))

        found_answer = False
        found_correct = False
        reason_str = ""
        raw_model_answer = None
        sanitized_model_answer = None

        for i in range(best_N_local):
            prediction_str = outputs[i]
            prediction_json = extract_first_complete_json(prediction_str)

            if prediction_json is None or "answer" not in prediction_json:
                prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            if prediction_json is None or "answer" not in prediction_json:
                try_extracted_answer = model_specific_extraction(model, prediction_str)
                if not try_extracted_answer:
                    continue
                else:
                    prediction_json = prediction_json or {}
                    prediction_json["answer"] = try_extracted_answer

            found_answer = True
            reason_str = prediction_json.get("reasoning", "")

            raw_model_answer = str(prediction_json["answer"])
            sanitized_model_answer = raw_model_answer.strip("'\"").replace('\n', '\\n')
            correct_answer = str(item["answer"]).strip("'\"")

            if sanitized_model_answer == correct_answer:
                solved_examples += 1
                found_correct = True
            break  # Stop after the first valid answer

        if not found_answer:
            no_answer += 1

        reason_lens.append(len(reason_str))

        parsed_item = item.copy()
        parsed_item["reasoning"] = reason_str
        parsed_item["model_answer"] = {
            "raw": raw_model_answer,
            "sanitized": sanitized_model_answer
        }
        parsed_item["correct_answer"] = {
            "raw": item["answer"],
            "sanitized": correct_answer
        }
        parsed_item["matched"] = found_correct
        parsed_results.append(parsed_item)

    acc = (solved_examples / num_total_examples * 100) if num_total_examples else 0
    no_ans_rate = (no_answer / num_total_examples * 100) if num_total_examples else 0
    avg_reason_len = (sum(reason_lens) / len(reason_lens)) if reason_lens else 0

    result = {
        "Model": model_name_replacement(model.split("%")[0]),
        "Mode": "first_answered",
        "Acc": f"{acc:.2f}",
        "No answer": f"{no_ans_rate:.2f}",
        "Total": num_total_examples,
        "Reason Lens": f"{avg_reason_len:.2f}",
    }
    return result, parsed_results


def gen_results(run_name_folders, data_name="crux", mode="eval", best_N=-1):
    """
    Generates and prints a table of results, plus writes a .summary.md and a .summary.json.
    Also saves parsed results to a parallel structure under 'result_dirs_parsed/'.
    """
    model_results = load_model_results(run_name_folders)
    columns = ["Model", "Mode", "Acc", "No answer", "Total", "Reason Lens"]
    rows = []

    for model_name, filepath in model_results.items():
        if mode == "best":
            result, parsed_results = eval_model_best(model_name, filepath, best_N)
        elif mode == "first_answered":
            result, parsed_results = eval_model_first_answered(model_name, filepath, best_N)
        else:
            result, parsed_results = eval_model(model_name, filepath)

        # Save parsed results to a parallel path in 'result_dirs_parsed'
        parsed_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        os.makedirs(os.path.dirname(parsed_filepath), exist_ok=True)
        with open(parsed_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)

        rows.append(result)

    # Sort rows by descending accuracy
    rows.sort(key=lambda x: -float(x["Acc"]))

    # Print table
    table_data = [[row[col] for col in columns] for row in rows]
    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))

    # Write to Markdown file
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(
            tabulate(table_data, headers=columns, tablefmt="github",
                                     stralign="center", numalign="center")
        )

    # Write to JSON file
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on CRUX-like dataset")
    parser.add_argument("--dataset", type=str, default="crux", help="Name of the dataset (default: crux)")
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["eval", "best", "first_answered"],
                        help="Evaluation mode: eval, best, or first_answered")
    parser.add_argument("--best_N", type=int, default=-1,
                        help="Number of outputs to consider for 'best' or 'first_answered' modes (-1 means all)")
    args = parser.parse_args()

    data_name = args.dataset
    mode = args.mode
    best_N = args.best_N

    # Configure your run_name_folders. Each key is "Model%Mode"
    # Values are the paths to JSON files with model outputs.
    run_name_folders = {
        "methods": f"result_dirs/{data_name}"
    }

    gen_results(run_name_folders, data_name=data_name, mode=mode, best_N=best_N)
