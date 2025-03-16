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
    """
    Calculate the confidence of the logits.
    logits: torch.Tensor, shape (batch_size, seq_length, vocab_size) or (seq_length, vocab_size)
    attention_mask: torch.Tensor, shape (batch_size, seq_length) or (seq_length)
    """
    logits = logits.contiguous()
    attention_mask = attention_mask.contiguous()
    V = logits.shape[-1]
    V_tensor = torch.tensor(V, dtype=logits.dtype, device=logits.device)
    logprob = torch.nn.functional.log_softmax(logits, dim=-1)
    conf = -1/V * torch.sum(logprob + torch.log(V_tensor), dim=-1)
    valid_conf = conf * attention_mask
    batch_confidence_list = (valid_conf.sum(dim=-1) / attention_mask.sum(dim=-1)).tolist()
    return batch_confidence_list

@torch.no_grad()
def confidence_with_file(filepath, output_file=None, batch_size=4):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    model_dir = data[0]["generator"]
    best_N = len(data[0]["output"])
    
    print("Loading model:", model_dir)
    torch.set_grad_enabled(False)
    
    # Use CUDA if available; the model is small so it fits on one GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding=True)
    
    # Add padding token if it is not already present.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.resize_token_embeddings(len(tokenizer))
        llm.embed_tokens = torch.nn.Embedding(
            llm.config.vocab_size, llm.config.hidden_size, padding_idx=llm.config.pad_token_id
        ).to(device)
        print("Added padding token to tokenizer")
        
    tokenizer.padding_side = "right"
    
    llm.eval()
    # Wrap with DataParallel if multiple GPUs are available.
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for faster inference (DataParallel).")
        llm = torch.nn.DataParallel(llm)
    
    llm.eval()
    print("Loaded model and tokenizer.")
    
    # Determine the output file path
    if output_file is None:
        output_file = os.path.splitext(filepath)[0] + f"-confidence-list.json"
    
    # Load already processed data if available.
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

        # Encode the input prompt.
        input_encoded = tokenizer(
            item["model_input"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = input_encoded['input_ids'].reshape(-1)
        input_attention_mask = input_encoded['attention_mask'].reshape(-1)
        input_length = input_attention_mask.sum().item()  # Actual token count of the prompt

        # Retrieve the top N outputs.
        outputs = item["output"][:best_N]
        
        # Classify outputs based on their raw text length (before tokenization).
        groups = {
            "small": {"outputs": [], "indices": []},
            "medium": {"outputs": [], "indices": []},
            "large": {"outputs": [], "indices": []}
        }
        for idx, text in enumerate(outputs):
            if len(text) > 10 * 1024:
                groups["large"]["outputs"].append(text)
                groups["large"]["indices"].append(idx)
            elif len(text) > 5 * 1024:
                groups["medium"]["outputs"].append(text)
                groups["medium"]["indices"].append(idx)
            else:
                groups["small"]["outputs"].append(text)
                groups["small"]["indices"].append(idx)
        
        # Prepare a list for the final confidence scores (in the original order).
        final_confidences = [None] * len(outputs)
        
        # Define batch sizes for each group.
        group_batch_sizes = {
            "small": batch_size,
            "medium": max(1, batch_size // 2),
            "large": max(1, batch_size // 4)
        }
        
        # Process each group separately.
        for group_name in ["small", "medium", "large"]:
            group_texts = groups[group_name]["outputs"]
            group_indices = groups[group_name]["indices"]
            if not group_texts:
                continue
            
            current_batch_size = group_batch_sizes[group_name]
            
            # Tokenize the outputs in this group.
            group_tokenized = tokenizer(
                group_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            group_outputs_ids = group_tokenized['input_ids']            # (n, seq_length_out)
            group_outputs_attention_mask = group_tokenized['attention_mask']  # (n, seq_length_out)
            
            # Build full sequences by concatenating the prompt and each output.
            full_ids_list = []
            full_attention_mask_list = []
            for i in range(group_outputs_ids.size(0)):
                combined_ids = torch.cat((input_ids, group_outputs_ids[i]), dim=0)
                combined_attention_mask = torch.cat((input_attention_mask, group_outputs_attention_mask[i]), dim=0)
                full_ids_list.append(combined_ids)
                full_attention_mask_list.append(combined_attention_mask)
            full_ids = torch.stack(full_ids_list)            # shape: (n, total_seq_length)
            full_attention_mask = torch.stack(full_attention_mask_list)  # shape: (n, total_seq_length)
            
            # Process logits in batches to avoid CUDA OOM.
            group_confidences = []
            num_batches = (full_ids.shape[0] + current_batch_size - 1) // current_batch_size
            for batch_idx in range(num_batches):
                torch.cuda.empty_cache()
                start_idx = batch_idx * current_batch_size
                end_idx = min((batch_idx + 1) * current_batch_size, full_ids.shape[0])
                batch_ids = full_ids[start_idx:end_idx].to(device)
                batch_attention_mask = full_attention_mask[start_idx:end_idx].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    batch_logits = llm(batch_ids, attention_mask=batch_attention_mask).logits.to('cpu')
                    batch_logits = batch_logits.to(torch.bfloat16)
                # Only consider logits for the output part (skip the prompt tokens).
                batch_outputs_logits = batch_logits[:, input_length:, :]
                # Use the output attention mask from the tokenized group (for this batch).
                batch_output_attention_mask = group_outputs_attention_mask[start_idx:end_idx]
                batch_confidence_list = confidence_logits(batch_outputs_logits, batch_output_attention_mask.cpu())
                group_confidences.extend(batch_confidence_list)
            
            # Place the computed confidences back in the correct (original) positions.
            for i, orig_idx in enumerate(group_indices):
                final_confidences[orig_idx] = group_confidences[i]
        
        if any(conf is None for conf in final_confidences):
            print(f"Warning: Some confidences were not computed for item at index {index}.")
        
        print("all_confidences:", final_confidences)
        
        # Save the confidence list with the current item.
        new_item["confidence_list"] = final_confidences
        new_item["processed_index"] = index
        to_write.append(new_item)
        
        # Write the updated list to a temporary file.
        try:
            with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(output_file)) as tmp_file:
                json.dump(to_write, tmp_file, indent=4)
                temp_name = tmp_file.name
            os.replace(temp_name, output_file)
        except Exception as e:
            print(f"Error writing to file: {e}")
            print("Exiting to prevent data loss.")
            break
        
        print(f"Processed {index + 1}/{total_items}")
        torch.cuda.empty_cache()

# Example usage:
# python3 script.py --input_file /path/to/input.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute confidence scores for model outputs using DataParallel for faster inference."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing.")
    args = parser.parse_args()
    
    confidence_with_file(args.input_file, args.output_file, args.batch_size)
