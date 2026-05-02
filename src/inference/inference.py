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

    # Unoptimized inference (important: no batching, no cache tuning yet)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=50, do_sample=True, temperature=0.8, top_p=0.95
        )

    # Decode output
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    text2 = tokenizer.decode(output[1], skip_special_tokens=True)

    print("\n--- OUTPUT ---")
    print(text)
    print(text2)


if __name__ == "__main__":
    main()
