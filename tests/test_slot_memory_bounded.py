"""Stress test: verifies that per-slot KV caches stay bounded under continuous load.

This test streams 50 requests through 4 slots, each generating 30 tokens.
After every step it inspects each slot's KV cache seq_len and asserts:

  1. No slot's seq_len ever exceeds its own (prompt_len + tokens_generated).
  2. Total KV cache memory stays below the theoretical maximum.
  3. All 50 requests complete successfully.

This would FAIL on the old shared-padded architecture because new requests
would inherit the max sequence length of surviving requests, causing
unbounded growth.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferstream.metrics import metrics
from inferstream.inference.engine import ContinuousBatchingEngine


def _load_model():
    """Shared model/tokenizer setup."""
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


def test_slot_memory_bounded() -> None:
    """Verify that per-slot KV caches never grow beyond each request's own length.

    Under the old architecture, a new request entering after others have been
    generating for a while would inherit the survivors' max seq_len via padding.
    Over continuous load this grows without bound.

    With per-slot KV caches, each slot's seq_len = prompt_len + tokens_generated,
    completely independent of other slots.
    """
    model, tokenizer, device = _load_model()
    metrics.reset()

    max_new_tokens = 30
    max_slots = 4
    num_requests = 50

    engine = ContinuousBatchingEngine(
        model=model, tokenizer=tokenizer, max_slots=max_slots
    )

    # Generate diverse prompts with varying lengths
    base_prompts = [
        "Once upon a time",
        "The quick brown fox jumps over the lazy dog in the park",
        "Hello world",
        "In the beginning there was nothing but darkness and then light appeared",
        "To be or not to be",
    ]
    prompts = [base_prompts[i % len(base_prompts)] for i in range(num_requests)]

    # Tokenize each prompt upfront so we know their true lengths
    prompt_token_lengths = {}
    for i, p in enumerate(prompts):
        req_id = engine.add_request(p, max_new_tokens=max_new_tokens)
        tokens = tokenizer(p, return_tensors="pt").input_ids
        prompt_token_lengths[req_id] = tokens.shape[1]

    print(f"\n{'='*70}")
    print(f"  SLOT MEMORY BOUNDED TEST — {num_requests} requests, {max_slots} slots")
    print(f"{'='*70}\n")

    completed_requests = []
    step_num = 0
    max_observed_seq_len = 0
    theoretical_max_seq_len = max(prompt_token_lengths.values()) + max_new_tokens

    while engine.has_pending_or_active():
        step_num += 1
        completed = engine.step()
        completed_requests.extend(completed)

        # ---------------------------------------------------------------
        # KEY ASSERTION: inspect each slot's KV cache seq_len
        # ---------------------------------------------------------------
        for slot in engine.slots:
            if slot.is_free:
                continue

            # The seq_len in the KV cache is tracked globally per-slot.
            kv_seq_len = engine.global_cache.slot_seq_lens[slot.slot_id]

            # The slot's KV cache covers everything EXCEPT the last generated token
            # (which will be cached during the next decode step).
            # So kv_seq_len + 1 must equal the expected total footprint.
            req = slot.request
            expected_total = prompt_token_lengths[req.request_id] + len(
                req.generated_tokens
            )
            assert kv_seq_len + 1 == expected_total, (
                f"Step {step_num}, slot {slot.slot_id}: "
                f"kv_seq_len + 1 ({kv_seq_len + 1}) != expected ({expected_total}) "
                f"[prompt={prompt_token_lengths[req.request_id]}, "
                f"generated={len(req.generated_tokens)}]"
            )

            # Track the maximum observed seq_len across all slots and steps
            max_observed_seq_len = max(max_observed_seq_len, kv_seq_len)

        # Log every 10 steps
        if step_num % 10 == 0:
            active_ids = [
                s.request.request_id for s in engine.slots if s.request is not None
            ]
            print(
                f"  Step {step_num:>3d} | active={active_ids} | "
                f"pending={len(engine.pending_requests)} | "
                f"completed={len(completed_requests)}/{num_requests}"
            )

    # ===================================================================
    # Final assertions
    # ===================================================================

    # All requests completed
    assert len(completed_requests) == num_requests, (
        f"Expected {num_requests} completions, got {len(completed_requests)}"
    )
    for req in completed_requests:
        assert req.status == "completed"
        assert len(req.generated_tokens) == max_new_tokens

    # The maximum observed seq_len must not exceed the theoretical max.
    # Under the old architecture, this would grow as
    # O(total_decode_steps_since_start) — much larger than theoretical_max.
    assert max_observed_seq_len <= theoretical_max_seq_len, (
        f"Max observed seq_len ({max_observed_seq_len}) exceeds theoretical max "
        f"({theoretical_max_seq_len}). This indicates sequence length inflation — "
        f"the bug we fixed."
    )

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Total requests completed:   {len(completed_requests)}")
    print(f"  Total steps:                {step_num}")
    print(f"  Max observed seq_len:       {max_observed_seq_len}")
    print(f"  Theoretical max seq_len:    {theoretical_max_seq_len}")
    print(f"  Slots used:                 {max_slots}")
    print()

    # Under the old architecture with 50 requests through 4 slots,
    # the shared seq_len would have grown to roughly:
    #   initial_max_prompt_len + (total_decode_steps)
    # where total_decode_steps ≈ (50/4) * 30 ≈ 375+ positions
    # instead of the correct max of ~45 (15 prompt tokens + 30 generated).
    print(
        f"  ✓ No sequence length inflation detected. "
        f"Each slot's KV cache stayed at its request's own length."
    )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_slot_memory_bounded()
