import json
import sys
import os
import re

sys.path.append("/data/xuandong_zhao/mnt/zheweikang/ZeroEval/src/evaluation") 

from eval_utils import extract_values_from_json, extract_first_complete_json, model_specific_extraction
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

unicode_to_latex = {
    'π': '\\pi',
    '√': '\\sqrt',
    '∞': '\\infty',
    '≤': '\\leq',
    '≥': '\\geq',
    '≠': '\\neq',
    '→': '\\to',
    '←': '\\leftarrow',
    '∑': '\\sum',
    '∫': '\\int',
    '∂': '\\partial',
    '∧': '\\land',
    '≈': '\\approx',
    '∈': '\\in',
    '⊂': '\\subset',
    '⊆': '\\subseteq',
    '⊇': '\\supseteq',
    '⊄': '\\not\\subset',
    '⊕': '\\oplus',
    '⊗': '\\otimes',
    '→': '\\rightarrow',
    '←': '\\leftarrow'
    # Add more symbols as needed
}

def unicode_to_latex_code(input_str):
    output_str = ""
    
    for char in input_str:
        if char in unicode_to_latex:
            output_str += unicode_to_latex[char]
        else:
            output_str += char
    
    return output_str


def is_amps_hard(data):
    if "subtask" not in data and "task" not in data:
        return False
    task_or_subtask = data["subtask"] if "subtask" in data.keys() else data["task"]
    if "amps_hard" in task_or_subtask:
        return True
    return False

def latex_to_fraction(latex_str):
    # Regular expression to match LaTeX fraction format
    pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
    
    # Replace LaTeX fraction with a regular fraction
    return re.sub(pattern, r'\1/\2', latex_str)

def sanitize_math_answers(answer):
    answer = str(answer).strip()
    # Ignore symbols like $
    answer = answer.replace("$", "").strip()
    # Remove "," in the number
    answer = answer.replace(",", "")
    
    try:
        answer = unicode_to_latex_code(answer)
        
        answer = strip_string(answer)
    except:
        pass
    
    if "frac" in answer:
        try:
            answer = latex_to_fraction(answer)
            answer = str(float(eval(answer)))
        except:
            pass
        
    # Convert to float and then to int if it's an integer
    try:
        num = float(answer)
        if num.is_integer():
            answer = str(int(num))
        else:
            answer = str(num)
    except:
        pass
    # Remove spaces
    answer = answer.replace(" ", "")
    return answer


def convert_AAAAA_to_A(answer):
    # convert AAAAA to A
    answer = str(answer)
    answer = answer.strip()
    if len(answer) >= 4 and answer[0] == answer[1] == answer[2] == answer[3]:
        return answer[0]
    return answer

def extract_answer_from_output(prediction_str, model, dataset = "math" ):
    prediction_json = extract_first_complete_json(prediction_str)
    flag_parsed_answer = True
    if prediction_json is None or "answer" not in prediction_json:
        prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
        # print("-")
    if prediction_json is None or "answer" not in prediction_json: 
        try_extracted_answer = model_specific_extraction(model, prediction_str)
        if try_extracted_answer:
            # print(f"Extracted answer from model: {try_extracted_answer}")
            prediction_json["answer"] = try_extracted_answer
        else:
            flag_parsed_answer = False 
    model_answer = None 
    if flag_parsed_answer:
        model_answer = str(prediction_json["answer"])
        # sanitize the answers
        model_answer = convert_AAAAA_to_A(model_answer)
        if "math" in dataset:
            model_answer = sanitize_math_answers(model_answer)
        if "crux" in dataset:
            model_answer = model_answer.strip("'\"").replace('\n', '\\n')
    return model_answer

def report_reason_length(prediction_str):
    prediction_json = extract_first_complete_json(prediction_str)
    if prediction_json is None or "reasoning" not in prediction_json:
        prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
    if prediction_json is None or "reasoning" not in prediction_json:
        return 0
    return len(prediction_json["reasoning"])