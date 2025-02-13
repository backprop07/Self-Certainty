import json
import re
from evaluation.eval_utils_padding import  extract_answer_from_output
import argparse

def extract_response_index(text: str) -> int:
    matches = re.findall(r"Response\s+(\d+)", text)
    if matches:
        return int(matches[-1])
    return None

def convert_json(input_file, output_file):
    """
    Converts the input JSON data to the desired format and writes it to the output file.
    
    Parameters:
    - input_file: Path to the input JSON file.
    - output_file: Path to the output JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    not_extracted = 0
    invalid_index = 0
    for item in data:
        index = extract_response_index(item["output"][0])
        if index is None:
            print(f"Could not extract response index")
            index = 0
            not_extracted += 1
        elif index not in range(len(item["responses"])):
            print(f"Invalid response index: {index}")
            index = 0
            invalid_index += 1
        item["output"] = [item["responses"][index]]
        converted_data.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4)
    
    print(f"Conversion complete. Output written to {output_file}")
    print(f"Number of items with response index not extracted: {not_extracted}")
    print(f"Number of items with invalid response index: {invalid_index}")
    
def convert_json_close(input_file, output_file, N=8):
    """
    Converts the input JSON data to the desired format and writes it to the output file.
    
    Parameters:
    - input_file: Path to the input JSON file.
    - output_file: Path to the output JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    not_extracted = 0
    invalid_index = 0
    for item in data:
        index = extract_response_index(item["output"][0])
        if index is None:
            print(f"Could not extract response index")
            index = 0
            not_extracted += 1
        elif index not in range(len(item["responses"])):
            print(f"Invalid response index: {index}")
            index = 0
            invalid_index += 1
        if extract_answer_from_output(item["responses"][index], "Llama3.1", "math") is None:
            for i in range(0, N):
                if extract_answer_from_output(item["responses"][i], "Llama3.1", "math") is not None:
                    index = i
                    break
            
        item["output"] = [item["responses"][index]]
        
        converted_data.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4)
    
    print(f"Conversion complete. Output written to {output_file}")
    print(f"Number of items with response index not extracted: {not_extracted}")
    print(f"Number of items with invalid response index: {invalid_index}")


if __name__ == "__main__":
    # Specify the input and output file paths
    # Use argparse to specify the input and output file paths
    parser = argparse.ArgumentParser(description='Convert JSON data to the desired format.')
    parser.add_argument('--input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('--dataset_type', choices=['open', 'close'], default='open', help='Type of dataset (open or close)')
    args = parser.parse_args()
    input_json_file = args.input_file    
    output_json_file = input_json_file.replace('.json', '-usc.json') 
    if args.dataset_type == 'open':
        convert_json(input_json_file, output_json_file)
    else:
        convert_json_close(input_json_file, output_json_file)
    
