import os
from typing import Dict, List

import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from inferstream.metrics import metrics


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int = 50,
) -> List[str]:
    """
    General inference function for causal language models.
    
    Args:
        model: The causal language model to use for generation.
        tokenizer: The tokenizer corresponding to the model.
        inputs: Dictionary of input tensors, containing at least 'input_ids' 
                and optionally 'attention_mask'.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        A list of decoded generated strings, stripped of special tokens.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    batch_size = input_ids.shape[0]
    input_length = input_ids.shape[1]

    metrics.observe("batch_size", batch_size)
    metrics.observe("input_tokens", input_length)
    past_key_values = None

    with torch.no_grad():
        metrics.start_timer("inference_total_time")

        for i in range(max_new_tokens):
            metrics.start_timer("token_generation_time")

            # For first iteration, use full input_ids + attention_mask
            # For subsequent iterations, use only the last token + growing attention_mask
            if i == 0:
                current_input_ids = input_ids
                current_attention_mask = attention_mask
            else:
                current_input_ids = next_token  # ONLY last token
                current_attention_mask = attention_mask  # still grows

            # print("Current input_ids shape:", current_input_ids.shape)
            # print("Current attention_mask shape:", current_attention_mask.shape)

            # 1. Forward pass
            # Plug in old cache if it exists
            outputs = model.forward(
                input_ids=current_input_ids, attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True  # Ensure caching is enabled
            )

            # 2. Get logits for LAST token only
            logits = outputs.logits[:, -1, :]  # shape: [B, vocab]

            # Get KV cache of all layers
            past_key_values = outputs.past_key_values
            
            if past_key_values is not None:
                cache_size_bytes = 0
                for layer_cache in past_key_values:
                    for item in layer_cache:
                        if isinstance(item, torch.Tensor):
                            cache_size_bytes += item.element_size() * item.nelement()
                metrics.observe("kv_cache_size_mb", cache_size_bytes / (1024 * 1024))

            # 3. Sample next token
            next_token = sample_next_token(logits, temperature=0.8, top_p=0.95)

            # 4. Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Append 1 to attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )

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
    output_strings: List[str] = []
    for output in input_ids:
        text = tokenizer.decode(output, skip_special_tokens=True)
        output_strings.append(text)

    return output_strings


def run_demo() -> None:
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


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9
) -> torch.Tensor:
    """
    Samples the next token from the output logits using nucleus sampling (top-p).
    
    Args:
        logits: Unnormalized log probabilities from the model.
        temperature: Controls the randomness of predictions. 
                     Lower values make it more deterministic.
        top_p: Nucleus sampling probability cutoff. Only the smallest set of 
               most probable tokens with probabilities that add up to top_p or higher 
               are kept for generation.
               
    Returns:
        A tensor containing the sampled token index.
    """
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
