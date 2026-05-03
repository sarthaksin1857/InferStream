import os
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferstream.metrics import metrics
from inferstream.inference.engine import ContinuousBatchingEngine


def _load_model():
    """Shared model/tokenizer setup for continuous batching tests."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model, tokenizer, device


def test_continuous_batching() -> None:
    """
    Demonstrates continuous batching with 15 prompts flowing through 8 slots.

    Prints a step-by-step log showing:
      - which slots are active each step
      - when requests complete and new ones are pulled from the queue
      - KV cache size staying bounded
    """
    model, tokenizer, device = _load_model()
    metrics.reset()

    process = psutil.Process(os.getpid())
    metrics.set_gauge(
        "memory_system_mb_start", process.memory_info().rss / (1024 * 1024)
    )
    if device.type == "mps":
        metrics.set_gauge(
            "memory_mps_mb_start",
            torch.mps.current_allocated_memory() / (1024 * 1024),
        )

    prompts = [
        "Once upon a time in distributed systems,",
        "What is the meaning of life?",
        "How do you build a large language model?",
        "To be or not to be, that is the question.",
        "The quick brown fox jumps over the lazy dog.",
        "A long time ago in a galaxy far, far away...",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago\u2014never mind how long precisely\u2014",
        "It is a truth universally acknowledged,",
        "In the beginning God created the heavens and the earth.",
        "Two roads diverged in a yellow wood,",
        "I wandered lonely as a cloud",
        "Water, water, everywhere, / And all the boards did shrink;",
        "Shall I compare thee to a summer's day?",
        "Fourscore and seven years ago our fathers brought forth",
    ]

    max_new_tokens = 20
    max_slots = 8

    engine = ContinuousBatchingEngine(
        model=model, tokenizer=tokenizer, max_slots=max_slots
    )

    for p in prompts:
        engine.add_request(p, max_new_tokens=max_new_tokens)

    print(f"\n{'='*70}")
    print(f"  CONTINUOUS BATCHING — {len(prompts)} prompts, {max_slots} slots")
    print(f"{'='*70}\n")

    metrics.start_timer("inference_total_time")

    completed_requests = []
    step_num = 0

    while engine.has_pending_or_active():
        step_num += 1

        active_ids_before = [r.request_id for r in engine.active_requests]
        pending_count = len(engine.pending_requests)

        completed = engine.step()
        completed_requests.extend(completed)

        active_ids_after = [r.request_id for r in engine.active_requests]

        # Build a readable step log
        completed_ids = [r.request_id for r in completed]
        new_ids = [rid for rid in active_ids_after if rid not in active_ids_before]

        log_parts = [f"Step {step_num:>3d}"]
        log_parts.append(f"active={active_ids_after}")
        log_parts.append(f"pending={pending_count - len(new_ids)}")
        if completed_ids:
            log_parts.append(f"  \u2714 completed: {completed_ids}")
        if new_ids:
            log_parts.append(f"  \u2192 new slots: {new_ids}")

        print(" | ".join(log_parts))

    total_duration = metrics.stop_timer("inference_total_time")
    total_tokens = sum(len(r.generated_tokens) for r in completed_requests)
    if total_duration > 0:
        metrics.observe("throughput_tokens_per_sec", total_tokens / total_duration)

    metrics.set_gauge(
        "memory_system_mb_end", process.memory_info().rss / (1024 * 1024)
    )
    if device.type == "mps":
        metrics.set_gauge(
            "memory_mps_mb_end",
            torch.mps.current_allocated_memory() / (1024 * 1024),
        )

    # Print completions
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    for req in sorted(completed_requests, key=lambda r: r.request_id):
        text = tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        print(f"\n--- Request {req.request_id} ---")
        print(f"  Prompt:     {req.prompt}")
        print(f"  Completion: {text}")

    print(f"\n{'='*70}")
    print("  METRICS")
    print(f"{'='*70}")
    metrics.print_summary()

    # Assertions to make this a real test
    assert len(completed_requests) == len(prompts), (
        f"Expected {len(prompts)} completions, got {len(completed_requests)}"
    )
    for req in completed_requests:
        assert req.status == "completed"
        assert len(req.generated_tokens) == max_new_tokens


if __name__ == "__main__":
    test_continuous_batching()
