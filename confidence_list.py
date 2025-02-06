import argparse
import json
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm

# Set the environment variable for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@torch.no_grad()
def confidence_logits(logits: torch.Tensor, attention_mask: torch.Tensor):
    '''
    Calculate the confidence of the logits
    logits: torch.tensor, shape (batch_size, seq_length, vocab_size) or (seq_length, vocab_size)
    attention_mask: torch.tensor, shape (batch_size, seq_length) or (seq_length)
    '''
    logits = logits.contiguous()
    attention_mask = attention_mask.contiguous()
    attention_mask = attention_mask.squeeze()
    V = logits.shape[-1] 
    logprob = torch.nn.functional.log_softmax(logits.view(-1, V), dim=-1)
    naive = torch.full_like(logprob, 1.0 / V)
    conf = torch.nn.functional.kl_div(logprob, naive, reduction='none').sum(dim=-1)
    conf = conf.view(-1)
    valid_conf = conf * attention_mask.view(-1)
    return valid_conf
    

@torch.no_grad()
def confidence_with_file(filepath, output_file=None, batch_size=8):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    model_dir = data[0]["generator"]
    best_N = len(data[0]["output"])
    
    print("Loading model: ", model_dir)
    torch.set_grad_enabled(False)
    llm = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    llm.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding=True)
    
    if "Llama" in model_dir:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.resize_token_embeddings(len(tokenizer))
        llm.embed_tokens = torch.nn.Embedding(
            llm.config.vocab_size, llm.config.hidden_size, padding_idx=llm.config.pad_token_id
        )
    
    print("Loaded")
    
    # Determine the output file path
    if output_file is None:
        output_file = os.path.splitext(filepath)[0] + f"-confidence-list.json"
    
    # Load existing output data if the file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f_out:
            try:
                to_write = json.load(f_out)
                print(f"Loaded {len(to_write)} already processed items.")
            except json.JSONDecodeError:
                print("Output file is corrupted or empty. Starting fresh.")
                to_write = []
    else:
        to_write = []
    
    total_items = len(data)
    processed_items = len(to_write)
    
    print(f"Total items to process: {total_items}. Already processed: {processed_items}.")
    
    for index in tqdm.tqdm(range(processed_items, total_items), desc="Processing inputs"):
        item = data[index]
        new_item = {k: v for k, v in item.items()}

        # Encode the input
        input_encoded = tokenizer(
            item["model_input"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = input_encoded['input_ids'].reshape(-1)
        input_attention_mask = input_encoded['attention_mask'].reshape(-1)
        input_length = input_attention_mask.sum().item()  # Actual token count

        # Retrieve the top N outputs
        outputs = item["output"][:best_N]
        outputs_encoded = tokenizer(
            outputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        outputs_ids = outputs_encoded['input_ids']  # Shape: (best_N, seq_length)
        outputs_attention_mask = outputs_encoded['attention_mask']  # Shape: (best_N, seq_length)

        # Calculate the actual length of each output (number of non-padded tokens)
        outputs_lengths = outputs_attention_mask.sum(dim=1)  # Shape: (best_N,)

        seq_length = outputs_ids.shape[1]
        # Reduce the bath size if the sequence length is too long
        if seq_length > 1024*8:
            batch_size = 4
        

        # This creates a tensor of shape (num_valid_outputs, input_length + output_length)
        full_ids = torch.stack([
            torch.cat((input_ids, output_ids), dim=0) for output_ids in outputs_ids
        ])
        
        full_attention_mask = torch.stack([
            torch.cat((input_attention_mask, output_attention_mask), dim=0) for output_attention_mask in outputs_attention_mask
        ])

        # Process logits in batches to avoid CUDA OOM
        confidence_list = []
        num_batches = (full_ids.shape[0] + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            torch.cuda.empty_cache()
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, full_ids.shape[0])
            batch_ids = full_ids[start_idx:end_idx]
            batch_attention_mask = full_attention_mask[start_idx:end_idx]
            batch_output_attention_mask = outputs_attention_mask[start_idx:end_idx]
            with torch.autocast(device_type="cuda", dtype=torch.float16):  # Use mixed precision
                batch_logits = llm(batch_ids, attention_mask=batch_attention_mask).logits.to('cpu')
            # Slice out the logits corresponding to the output tokens
            batch_outputs_logits = batch_logits[:, input_length:, :]  # Shape: (batch_size, output_length, vocab_size)
            batch_confidences = confidence_logits(batch_outputs_logits, batch_output_attention_mask)
            confidence_list.append(batch_confidences)

        all_confidences = torch.cat(confidence_list, dim=0).view(-1, seq_length).sum(dim=1).view(-1)/outputs_lengths
        all_confidences = all_confidences.tolist()
        print("all_confidences:", all_confidences)

        # Assign the confidence list to the new item
        new_item["confidence_list"] = all_confidences
        new_item["processed_index"] = index

        to_write.append(new_item)

        # Write the updated list to a temporary file
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(output_file)) as tmp_file:
                json.dump(to_write, tmp_file, indent=4)
                temp_name = tmp_file.name
            # Atomically replace the original file with the temporary file
            os.replace(temp_name, output_file)
        except Exception as e:
            print(f"Error writing to file: {e}")
            print("Exiting to prevent data loss.")
            break

        print(f"Processed {index + 1}/{total_items}")

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()


# Example usage:
# python3 src/confidence_list.py --input_file /path/to/input.json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute confidence scores for model outputs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output_file
    batch_size = args.batch_size
    
    confidence_with_file(input_file, output_file, batch_size)
    
    