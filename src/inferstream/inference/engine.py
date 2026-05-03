import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferstream.metrics import metrics

def run_demo() -> None:
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

    batch_size = input_ids.shape[0]
    input_length = input_ids.shape[1]
    metrics.observe("batch_size", batch_size)
    metrics.observe("input_tokens", input_length)

    max_new_tokens = 50

    with torch.no_grad():
        metrics.start_timer("inference_total_time")

        for i in range(max_new_tokens):
            metrics.start_timer("token_generation_time")

            # 1. Forward pass
            outputs = model.forward(
                input_ids=input_ids, attention_mask=(inputs["attention_mask"])
            )

            # 2. Get logits for LAST token only
            logits = outputs.logits[:, -1, :]  # shape: [B, vocab]

            # 3. Sample next token
            next_token = sample_next_token(logits, temperature=0.8, top_p=0.95)

            # 4. Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            duration = metrics.stop_timer("token_generation_time")
            if i == 0:
                metrics.observe("time_to_first_token_ms", duration * 1000)
            else:
                metrics.observe("time_per_output_token_ms", duration * 1000)

            metrics.inc("total_generated_tokens", batch_size)

        total_duration = metrics.stop_timer("inference_total_time")
        if total_duration > 0:
            metrics.observe(
                "throughput_tokens_per_sec",
                (max_new_tokens * batch_size) / total_duration,
            )

    # Decode
    for i, output in enumerate(input_ids):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\n--- OUTPUT {i} ---")
        print(text)

    # Print collected metrics
    metrics.print_summary()


def sample_next_token(logits, temperature=1.0, top_p=0.9):
    # Apply temperature
    logits = logits / temperature

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Top-p (nucleus sampling)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumulative_probs > top_p

    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_probs[cutoff] = 0

    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(sorted_probs, num_samples=1)

    next_token = torch.gather(sorted_indices, -1, next_token)

    return next_token
