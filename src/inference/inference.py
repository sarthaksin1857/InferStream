import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # Device (Mac MPS or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Small, fast model (good starter choice)
    model_name = "distilgpt2"

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    # Input prompt
    prompt = "Once upon a time in distributed systems,"
    prompt2 = "What is the meaning of life?"

    # Tokenize as a tensor in pytorch format
    # Move it to the memory and mark it as mps compatabile
    inputs = tokenizer(
        [prompt, prompt2],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    input_ids = inputs["input_ids"]

    max_new_tokens = 50

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # input_ids_list = input_ids.tolist()

            # for i, batch in enumerate(input_ids_list):
            #     print(f"Batch {i}: {batch}")

            # 1. Forward pass
            outputs = model.forward(
                input_ids=input_ids, attention_mask=(inputs["attention_mask"])
            )

            # Outputs = Batch size, sequence length, vocab size
            # We dont care what comes after any sequence but the next word, so we will only look at the last token in the sequence for each batch item
            # Colon in python means take everything in this dimension and -1 means take the last item in that dimension, so we are taking the last token for each batch item and all vocab logits for that token

            # 2. Get logits for LAST token only
            logits = outputs.logits[:, -1, :]  # shape: [B, vocab]

            # print(
            #     f"Logits shape: {logits.shape}"
            # )  # [2,50257] since we have two prompts and vocab size of 50257 for distilgpt2

            # 3. Sample next token
            next_token = sample_next_token(logits, temperature=0.8, top_p=0.95)

            # 4. Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode
    for i, output in enumerate(input_ids):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\n--- OUTPUT {i} ---")
        print(text)


def sample_next_token(logits, temperature=1.0, top_p=0.9):
    # Apply temperature
    # Temperature controls how sharply the model prefers the top tokens
    # Super small and we will have deterministic output
    logits = logits / temperature

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Top-p (nucleus sampling)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)


    # Sum all probabilities as a tensor (final entry should be 1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # print(f"Cumulative probs: {cumulative_probs.shape}")

    # Remove tokens with cumulative prob > top_p
    # If you pick the hightest token we will have deterministic output
    # Greater than on a tensor returns a tensor of booleans, so we will have a boolean mask of which tokens to remove
    cutoff = cumulative_probs > top_p

    # False → allowed token
    # True  → remove token
    # We want to keep the first token that exceeds the top_p threshold, so we shift the cutoff mask to the right by one position and set the first position to False
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    # Both are tensors
    # Doing [] on a tensor with a boolean mask will set the values where the mask is True to 0, so we are setting the probabilities of the tokens we want to remove to 0
    # Ie only edit the values which are true
    sorted_probs[cutoff] = 0

    # Normalize probabilities after nuking some of them
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample (weighted random choice) from the filtered distribution
    # We are only running this sample once. We could do it like 5 times
    # Most likely to be 0 since thats the highest probability
    next_token = torch.multinomial(sorted_probs, num_samples=1)

    # Map back to original indices
    # Take 0 -> turn it into actual index in the vocab
    next_token = torch.gather(sorted_indices, -1, next_token)

    return next_token


if __name__ == "__main__":
    main()
