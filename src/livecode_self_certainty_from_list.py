import json
import re
import argparse
import os

def extract_code(output_text):
    """
    Extracts Python code snippets from the given output text.
    Looks for code blocks enclosed within ```python and ```.

    Parameters:
    - output_text (str): The text containing code snippets.

    Returns:
    - str: A single string containing all extracted Python code snippets concatenated together.
           Returns an empty string if no code snippets are found.
    """
    # Regular expression pattern to find ```python code blocks
    pattern = re.compile(r'```python\s*\n(.*?)\n```', re.DOTALL)
    matches = pattern.findall(output_text)
    
    if not matches:
        # No code blocks found
        return ""
    
    # Strip leading/trailing whitespace from each code snippet
    stripped_matches = [match.strip() for match in matches]
    
    concatenated_code = "\n\n".join(stripped_matches)
    
    return concatenated_code

def convert_json(input_file, output_file, best_N=8):
    """
    Converts the input JSON data to the desired format and writes it to the output file.
    
    Parameters:
    - input_file: Path to the input JSON file.
    - output_file: Path to the output JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    for item in data:
        question_id = item.get('question_id', 'unknown_id')
        code_list = []
        outputs = item.get('output', [])[0:best_N]
        confidence_list = item.get('confidence_list', [])[0:best_N]
        best_index = confidence_list.index(max(confidence_list))
        code_list.append(extract_code(outputs[best_index]))
        converted_data.append({
            "question_id": question_id,
            "code_list": code_list
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4)
    
    print(f"Conversion complete. Output written to {output_file}")

if __name__ == "__main__":
    # Specify the input and output file paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--best_N", type=int, default=16)
    args = parser.parse_args()
    
    input_json_file = args.input_file
    if args.output_file is None:
        os.makedirs("eval", exist_ok=True)
        output_json_file = 'eval/'+input_json_file.split('/')[-1].replace(".json", f"-confidence-{args.best_N}.json")
    else:
        output_json_file = args.output_file
    convert_json(input_json_file, output_json_file, best_N=args.best_N)
