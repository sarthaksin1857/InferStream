import os
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferstream.metrics import metrics
from inferstream.inference.engine import ContinuousBatchingEngine

def test_run_demo() -> None:
    """
    Runs a standalone demo of the inference engine.
    
    Loads a small language model (distilgpt2), tracks memory usage,
    runs the inference loop using the ContinuousBatchingEngine,
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

    # Input prompts
    prompt = "Once upon a time in distributed systems,"
    prompt2 = "What is the meaning of life?"

    max_new_tokens = 50

    engine = ContinuousBatchingEngine(model=model, tokenizer=tokenizer, max_slots=8)
    engine.add_request(prompt, max_new_tokens=max_new_tokens)
    engine.add_request(prompt2, max_new_tokens=max_new_tokens)

    completed_requests = []
    while engine.has_pending_or_active():
        completed = engine.step()
        completed_requests.extend(completed)

    metrics.set_gauge("memory_system_mb_end", process.memory_info().rss / (1024 * 1024))
    if device.type == "mps":
        metrics.set_gauge(
            "memory_mps_mb_end", torch.mps.current_allocated_memory() / (1024 * 1024)
        )

    for req in sorted(completed_requests, key=lambda r: r.request_id):
        print(f"\n--- OUTPUT {req.request_id} ---")
        text = tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        print(f"Prompt: {req.prompt}")
        print(f"Completion: {text}")

    # Print collected metrics
    metrics.print_summary()

if __name__ == "__main__":
    test_run_demo()
