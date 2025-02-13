# Self-Certainty Evaluation

This repository provides tools for evaluating **self-certainty**, a metric designed to measure model confidence, as an extension of the [ZeroEval](https://github.com/WildEval/ZeroEval) project.

The self-certainty metric is calculated using the following formula:\
**Self-Certainty**
```math
-\frac{1}{nV} \sum_{i=1}^n \sum_{j=1}^{V} \log \left( V \cdot p(j|x , y_{&lti} ) \right)
```
Where:

- $n$ = Number of tokens in one sentence.
- $V$ = Vocabulary size.
- $p(j|x, y_{<i})$ = Probability of token \( j \) given the context \( x \) and previous tokens $y_{<i}$.

## Installation

Ensure you have [SymPy](https://www.sympy.org/) installed. You can install it via:

```bash
pip install sympy
```

### Integrating Self-Certainty with ZeroEval

To integrate the Self-Certainty extension with the ZeroEval project, follow these steps:

1. Clone this repository.
2. Copy the necessary files into the appropriate directories using the following command:

```bash
cp -r ZeroEval_padding/src/* ZeroEval/src/
```

## Usage

### Self-certainty Calculation (`confidence_list.py`)

Calculate the self-certainty score for a list of outputs based on the given input.

**Example usage:**

```bash
python3 src/confidence_list.py --input_file /path/to/input.json
```

The input file should be a JSON file containing:

- `"generator"`: Path to the model used for generation.
- `"output"`: List of model responses.
- `"input"`: The string specifying the model input.

By default, the self-certainty list will be written to `/path/to/input-confidence-list.json`.

### Choose Answer with Highest Self-Certainty Score
#### For Fixed Answer Questions (`self_certainty_from_list.py`)

This script selects the answer with the highest self-certainty score from a list of outputs. Answers without extractable content are assigned a confidence score of -infinity.

**Example usage:**

```bash
python3 src/self_certainty_from_list.py --input_file /path/to/input.json --best_N 16
```
#### For Code Generation (`livecode_self_certainty_from_list.py`)

This script selects the answer with the highest confidence score and parses it into the LiveCode format (`{"question_id", "code_list"}`).

**Example usage:**

```bash
python3 src/livecode_self_certainty_from_list.py --input_file /path/to/input.json --output_file /path/to/output.json --best_N 16
```

### Borda Voting on Output List (`voting_from_list.py`)

Performs Borda voting on a list of outputs. The majority vote is equivalent to Borda voting with \( p = 0 \). This is supported for fixed-answer questions only.

**Example usage:**

```bash
python3 src/voting_from_list.py --input_file /path/to/input.json --best_N 16 --power 0.5
```

### LiveCode Parsing (`livecode_parsing.py`)

This script parses the first item in the output list of a JSON file into the LiveCode format.

**Example usage:**

```bash
python3 src/livecode_parsing.py --input_file /path/to/input.json --output_file /path/to/output.json
```

### USC Generation

For USC generation, modify the dataset name (e.g., "gsm") in ZeroEval generation to `"usc-N-path/to/file.json"`, where \( N \) is the number of samples per question to be considered.

**Example usage:**

```bash
bash zero_eval_local.sh -d "usc-8-path/to/samples.json" -m model_path -p model-usc -s 2 -b 4
```

### USC Selection from Outputs (`usc_from_outputs.py`)

The `usc_from_outputs.py` script assists USC in selecting a specific output index. When `--dataset_type` is set to `close`, it helps USC choose the first extractable answer if the original answer is not extractable.

## Attributions

This project builds upon the following open-source repositories:

### ZeroEval

- **Repository:** [ZeroEval](https://github.com/WildEval/ZeroEval) **License:** [Apache License 2.0](https://github.com/WildEval/ZeroEval/blob/main/LICENSE)
- **Description:** A unified framework for evaluating instruction-tuned large language models on tasks like MMLU and GSM.

### LiveBench

- **Repository:** [LiveBench](https://github.com/LiveBench/LiveBench) **License:** [Apache License 2.0](https://github.com/LiveBench/LiveBench/blob/main/LICENSE)
- **Description:** A challenging, continuously updated benchmark that sources new questions monthly from various contemporary datasets.
