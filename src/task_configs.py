from datasets import load_dataset
from _TEMPLATES import apply_mc_template, apply_lgp_grid_template, apply_oeqa_template
from evaluation.livecode_utils import load_code_generation_dataset, get_generic_question_template_answer, CodeGenerationProblem, extract_question
import json
from templates.USC import usc_prompt
from evaluation.eval_utils_padding import extract_answer_from_output

def load_livebench_data():
    dataset = load_dataset("livebench/math", split="test")
    dataset = dataset.map(lambda x, idx: {'id': idx}, with_indices=True)
    purge_ending = lambda x: x[0]
    dataset = dataset.map(lambda x: {'problem': purge_ending(x['turns'])})
    dataset = dataset.map(lambda x: {'answer': x['ground_truth']})
    dataset = dataset.select_columns(['problem', 'answer', 'id', "task", "subtask"])
    return dataset
        
        
def mapping_task_names(data_name):
    """
    Mapping the task names to the dataset and id name.
    """
    id_name = "id"
    if data_name == "mmlu-redux":
        dataset = load_dataset("yuchenlin/zero-eval", "mmlu-redux", split="test")
    elif data_name == "gsm":
        dataset = load_dataset("yuchenlin/zero-eval", "gsm", split="test")
    elif data_name == "zebra-grid":
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
    elif data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")  
    elif data_name == "numersense-v2":
        dataset = load_dataset("yuchenlin/zero-eval", "numersense-v2", split="test")
    elif data_name == "crux":
        dataset = load_dataset("flydust/zero-eval", "crux", split="test")
    elif data_name == "math-l5":
        dataset = load_dataset("AI-MO/aimo-validation-math-level-5", split="test")
    elif data_name == "math-lx-new":
        path = "path/to/math-lx.json"
        dataset = load_dataset("json", data_files=path, split="test")
    elif data_name == "livecode":
        dataset = load_code_generation_dataset()  
    elif data_name == "livebench-math":
        dataset = load_livebench_data()
    elif data_name.startswith("usc-"):
        _, str_best_N, json_path = data_name.split("-", 2)
        best_N = int(str_best_N)
        dataset = json.load(open(json_path, "r"))
        for i in range(len(dataset)):
            dataset[i]["responses"] = dataset[i]["output"][:best_N]
            del dataset[i]["output"]
    else:
        raise ValueError(f"Data name {data_name} not supported")
    
    return dataset, id_name

def prompt_generation(data_name, data_item, args):
    """
    Generate prompt for different tasks.
    """
    if data_name in ["mmlu-redux"]:  # and other multiple-choice QA dataset 
        prompt = apply_mc_template(data_item) 
    elif data_name in ["alpaca_eval"]:
        prompt = data_item["instruction"]
    elif data_name in ["zebra-grid"]:
        prompt = apply_lgp_grid_template(data_item) 
    elif data_name in ["gsm"] or "math" in data_name:
        question_key = "question"
        if data_name == "math-l5" or data_name == "livebench-math" or ("math" in data_name):
            question_key = "problem"
        prompt = apply_oeqa_template(data_item, question_key = question_key)
    elif data_name in ["crux"]:
        prompt = apply_oeqa_template(data_item, cot=True) # cot?
    elif data_name in ["livecode"]:
        question_key = "question"
        data_formatted = CodeGenerationProblem(**data_item)
        prompt = get_generic_question_template_answer(data_formatted)
    elif data_name in ["numersense-v2"]:
        if "no_cot" in args.run_name:
            prompt = apply_oeqa_template(data_item, cot=False)
        prompt = apply_oeqa_template(data_item)
    elif "usc-" in data_name:
        question = extract_question(data_item["chat_history"][0])
        outputs = data_item["responses"]
        prompt = usc_prompt(question, outputs)
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return prompt

def result_format(output_item, args):
    """
    Modify the output format for different tasks if needed.
    """
    if args.data_name in ["alpaca_eval"]:
        output_item["output"] = output_item["output"][0] # use str instead of list 
    elif args.data_name in ["zebra-grid"]:
        if "solution" in output_item:
            del output_item["solution"]
    elif args.data_name in ["livecode"]:
        if "question_content" in output_item:
            del output_item["question_content"]
        if "public_test_cases" in output_item:
            del output_item["public_test_cases"]
        if "private_test_cases" in output_item:
            del output_item["private_test_cases"]
    else:
        pass 
    return output_item
