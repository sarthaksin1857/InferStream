import os
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferstream.metrics import metrics
from inferstream.inference.engine import generate

def test_run_demo() -> None:
    """
    Runs a standalone demo of the inference engine.
    
    Loads a small language model (distilgpt2), tracks memory usage,
    runs the inference loop using the generic generate function,
    and reports latency/throughput metrics.
    """
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

    process = psutil.Process(os.getpid())
    metrics.set_gauge(
        "memory_system_mb_start", process.memory_info().rss / (1024 * 1024)
    )
    if device.type == "mps":
        metrics.set_gauge(
            "memory_mps_mb_start", torch.mps.current_allocated_memory() / (1024 * 1024)
        )

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

    max_new_tokens = 50

    output_strings = generate(
        model=model,
        tokenizer=tokenizer,
        inputs=dict(inputs),
        max_new_tokens=max_new_tokens,
    )

    metrics.set_gauge("memory_system_mb_end", process.memory_info().rss / (1024 * 1024))
    if device.type == "mps":
        metrics.set_gauge(
            "memory_mps_mb_end", torch.mps.current_allocated_memory() / (1024 * 1024)
        )

    for i, text in enumerate(output_strings):
        print(f"\n--- OUTPUT {i} ---")
        print(text)

    # Print collected metrics
    metrics.print_summary()

if __name__ == "__main__":
    test_run_demo()
