# Self-Certainty Evaluation

This repository provides codes of our paper [Scalable Best-of-N Selection for Large Language Models via Self-Certainty](https://arxiv.org/abs/2502.18581), in which we propose self-certainty, a metric designed to measure model confidence.

**Self-Certainty** is calculated using the following formula:

```math
-\frac{1}{nV} \sum_{i=1}^n \sum_{j=1}^{V} \log \left( V \cdot p(j|x , y_{&lti} ) \right)
```
Where:

- $n$ = Number of tokens in one sentence.
- $V$ = Vocabulary size.
- $p(j|x, y_{<i})$ = Probability of token $j$ given the context $x$ and previous tokens $y_{<i}$.

For more details, please refer to our paper. And if you find this work useful, please consider citing our paper:
```
@article{kang2025scalable,
  title={Scalable Best-of-N Selection for Large Language Models via Self-Certainty},
  author={Kang, Zhewei and Zhao, Xuandong and Song, Dawn},
  journal={arXiv preprint arXiv:2502.18581},
  year={2025}
}
```

## Installation
Ensure you have [SymPy](https://www.sympy.org/) installed. You can install it via:

```bash
pip install sympy
```

**Integrating Self-Certainty with ZeroEval**

The code is an extension of the [ZeroEval](https://github.com/WildEval/ZeroEval) project.
To integrate the Self-Certainty extension with the ZeroEval project, follow these steps:

1. Clone this repository.
2. Copy the necessary files into the appropriate directories using the following command:

```bash
cp -r Self-Certainty/src/* ZeroEval/src/
```

## Usage

### 1. Self-Certainty Calculation

This script computes a self-certainty score for a collection of model outputs based on a given input. The implementation is provided in `confidence_list.py`.

**Example Usage:**

```bash
python3 src/confidence_list.py --input_file /path/to/input.json
```

**Input File Requirements:**

The JSON input file must include the following keys:
- **"generator"** (optional): The path to the model used for generating responses.
- **"output"**: An array containing the model’s responses.
- **"input"**: The text input provided to the model.

**Output Details:**

By default, the script writes the self-certainty scores to a file named `/path/to/input-confidence-list.json`. If the `--model_dir` option is not specified, the script defaults to using the `generator` value from the first entry in the input file.

### 2. Choose Answer with Highest Self-Certainty Score

**For Fixed Answer Questions**

The `self_certainty_from_list.py` script selects the answer with the highest self-certainty score from a list of outputs. Answers without extractable content are assigned a confidence score of -infinity.

*Example usage:*

```bash
python3 src/self_certainty_from_list.py --input_file /path/to/input.json --best_N 16
```

**For Code Generation** 

The `livecode_self_certainty_from_list.py` script selects the answer with the highest confidence score and parses it into the LiveCode format (`{"question_id", "code_list"}`).

*Example usage:*

```bash
python3 src/livecode_self_certainty_from_list.py --input_file /path/to/input.json --output_file /path/to/output.json --best_N 16
```

### 3. Borda Voting on Output List 

The `voting_from_list.py` script performs Borda voting on a list of outputs. The majority vote is equivalent to Borda voting with \( p = 0 \). This is supported for fixed-answer questions only.

*Example usage:*

```bash
python3 src/voting_from_list.py --input_file /path/to/input.json --best_N 16 --power 0.5
```

### 4. LiveCode Parsing 

The `livecode_parsing.py` script parses the first item in the output list of a JSON file into the LiveCode format.

*Example usage:*

```bash
python3 src/livecode_parsing.py --input_file /path/to/input.json --output_file /path/to/output.json
```

### 5. Universal Self-Consistency (USC) Generation

For USC generation, modify the dataset name (e.g., "gsm") in ZeroEval generation to `"usc-N-path/to/file.json"`, where \( N \) is the number of samples per question to be considered.

*Example usage:*

```bash
bash zero_eval_local.sh -d "usc-8-path/to/samples.json" -m model_path -p model-usc -s 2 -b 4
```

### 6. USC Selection from Outputs

The `usc_from_outputs.py` script assists USC in selecting a specific output index. When `--dataset_type` is set to `close`, it helps USC choose the first extractable answer if the original answer is not extractable.

## References

This project builds upon the following open-source repositories:

ZeroEval

- *Repository:* [ZeroEval](https://github.com/WildEval/ZeroEval) *License:* [Apache License 2.0](https://github.com/WildEval/ZeroEval/blob/main/LICENSE)
- *Description:* A unified framework for evaluating instruction-tuned large language models on tasks like MMLU and GSM.

LiveBench

- *Repository:* [LiveBench](https://github.com/LiveBench/LiveBench) *License:* [Apache License 2.0](https://github.com/LiveBench/LiveBench/blob/main/LICENSE)
- *Description:* A challenging, continuously updated benchmark that sources new questions monthly from various contemporary datasets.
