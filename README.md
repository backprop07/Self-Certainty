# Self-Certainty

This repository is used for self-certainty evaluation as an extension of the [ZeroEval](https://github.com/WildEval/ZeroEval) project.

## Installation

Ensure you have [SymPy](https://www.sympy.org/) installed. You can install it via:

```bash
pip install sympy
```

### Integrating ZeroEval_padding with ZeroEval

After downloading the ZeroEval project, follow these steps to integrate the ZeroEval_padding extension:

1. Clone this repository.
2. Copy the necessary files to their appropriate directories using the following command:

```bash
cp -r ZeroEval_padding/src/* ZeroEval/src/
```

## Usage

### Self-certainty calculation

`confidence_list.py`: Calculates the self-certainty score for a list of outputs based on the given input.

**Example usage:**

```bash
python3 src/confidence_list.py --input_file /path/to/input.json
```

The input file should be a JSON file with each item containing:

- `"generator"`: Path to the model used for generation.
- `"output"`: List of model responses.
- `"input"`: The string specifying the model input.

By default, this will write the self-certainty list to `/path/to/input-confidence-list.json`.

### `self_certainty_from_list.py`

Chooses the answer with the highest confidence score from a list of outputs, and assigns -infinity to answers without extractable content.

**Example usage:**

```bash
python3 src/self_certainty_from_list.py --input_file /path/to/input.json --best_N 16
```

### `voting_from_list.py`

Performs Borda voting on a list of outputs. The majority vote is equivalent to Borda voting with \( p = 0 \).

**Example usage:**

```bash
python3 src/voting_from_list.py --input_file /path/to/input.json --best_N 16 --power 0.5
```

### `livecode_self_certainty_from_list.py`

Chooses the answer with the highest confidence score from the list of outputs, and parses the answer into the LiveCode format (`{"question_id", "code_list"}`).

**Example usage:**

```bash
python3 src/livecode_self_certainty_from_list.py --input_file /path/to/input.json --output_file /path/to/output.json --best_N 16
```

### `livecode_parsing.py`

Parses the first item in the outputs to the LiveCode format.

**Example usage:**

```bash
python3 src/livecode_parsing.py --input_file /path/to/input.json --output_file /path/to/output.json
```

### `usc_from_outputs.py`

Helps USC choose the specified index of outputs. When `--dataset_type` is set to `close`, it helps USC select the first extractable answer if the original answer is not extractable.

### `task_configs.py`

To support evaluation for the MATH dataset (in addition to the original math-l5 dataset in ZeroEval), modify the `math-lx` section to specify the path to the extracted dataset.

For USC generation, replace the dataset name (e.g., "gsm") with `"usc-N-path/to/file.json"`, where N is the number of samples per question to consider.

**Example usage:**

```bash
bash zero_eval_local.sh -d "usc-8-path/to/samples.json" -m model_path -p model-usc -s 2 -b 4
```

### `evaluation/new_math.py`

Extends the `math_eval.py` in ZeroEval to support evaluation for the LiveBench-Math dataset and adds methods like "first_answered" and "best."

### `evaluation/new_crux.py`

Extends `crux_eval.py` to support the "first_answered" and "best" evaluation methods.

**Example usage:**

```bash
python3 src/evaluation/new_crux.py --dataset /path/to/input.json --mode best --best_N 16
```

## Attributions

This project incorporates and builds upon the following open-source repositories:

### ZeroEval

- **Repository:** [ZeroEval](https://github.com/WildEval/ZeroEval) **License:** [Apache License 2.0](https://github.com/WildEval/ZeroEval/blob/main/LICENSE)
- **Description:** A unified framework for evaluating instruction-tuned large language models on tasks like MMLU and GSM.

### LiveBench

- **Repository:** [LiveBench](https://github.com/LiveBench/LiveBench) **License:** [Apache License 2.0](https://github.com/LiveBench/LiveBench/blob/main/LICENSE)
- **Description:** A challenging, continuously updated benchmark that sources new questions monthly from various contemporary datasets.
