import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import psutil
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from inferstream.metrics import metrics


# ---------------------------------------------------------------------------
# Request: the unit of work flowing through the engine
# ---------------------------------------------------------------------------
# Each Request tracks a single prompt from submission through completion.
# The coordinator assigns a string request_id; internally the engine also
# assigns an integer slot id so we can index into batch tensors.
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """A single inference request managed by the ContinuousBatchingEngine.

    Fields:
        request_id:       Unique identifier (int slot id assigned by the engine).
        prompt:           The raw text prompt submitted by the caller.
        input_ids:        Tokenized prompt tensor, shape [1, seq_len].
        max_new_tokens:   How many tokens to generate before marking complete.
        generated_tokens: Token ids produced so far (appended each step).
        status:           Lifecycle state — "pending" → "active" → "completed".
        external_id:      Optional string id from an external system (e.g. the
                          coordinator's request_id) so callers can correlate
                          completed Requests back to their own tracking.
    """
    request_id: int
    prompt: str
    input_ids: torch.Tensor
    max_new_tokens: int
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "pending"  # pending, active, completed
    external_id: Optional[str] = None


# ---------------------------------------------------------------------------
# ContinuousBatchingEngine
# ---------------------------------------------------------------------------
# This is the core of the inference server.  Instead of waiting for a full
# batch to finish before accepting new work (static batching), we operate
# at *iteration-level* granularity:
#
#   1. Each call to step() generates exactly ONE new token for every active
#      slot in the batch.
#   2. When a slot finishes (hit max_new_tokens), it is immediately evicted
#      and a pending request takes its place on the *next* step().
#   3. The maximum number of concurrent slots (max_slots) puts a hard cap
#      on KV-cache memory, making peak usage predictable regardless of how
#      many requests are queued.
#
# Lifecycle of a request through the engine:
#
#   add_request()          →  enqueued in self.pending_requests
#   step() picks it up     →  PREFILL forward pass (full prompt), first token
#                              generated, KV cache spliced into the batch
#   subsequent step() calls→  DECODE forward pass (1 token), appended to
#                              generated_tokens until max_new_tokens reached
#   step() evicts it       →  returned in the completed list, KV cache slice
#                              removed from the batch
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """Iteration-level continuous batching engine.

    Args:
        model:      A HuggingFace causal LM already on the target device.
        tokenizer:  The corresponding tokenizer.
        max_slots:  Maximum number of requests generating tokens at once.
                    This directly caps KV-cache memory usage.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_slots: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_slots = max_slots

        # Queues
        self.pending_requests: List[Request] = []
        self.active_requests: List[Request] = []

        # Auto-incrementing slot id counter
        self.next_request_id = 0

        # Shared KV-cache and attention mask for the active batch.
        # Both are None when no requests are active.
        #   past_key_values: DynamicCache — per-layer (keys, values) tensors
        #                    with shape [batch, heads, seq_len, head_dim]
        #   attention_mask:  [batch, seq_len] — 1 for real tokens, 0 for padding
        self.past_key_values = None
        self.attention_mask = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        external_id: Optional[str] = None,
    ) -> int:
        """Enqueue a new prompt for generation.

        The prompt is tokenized immediately so we pay that cost upfront,
        but no forward pass happens until step() pulls it into an active slot.

        Returns:
            The engine-internal request_id (int).
        """
        req_id = self.next_request_id
        self.next_request_id += 1

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )
        req = Request(
            request_id=req_id,
            prompt=prompt,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            external_id=external_id,
        )
        self.pending_requests.append(req)
        return req_id

    def has_pending_or_active(self) -> bool:
        """True while there is still work to do."""
        return len(self.pending_requests) > 0 or len(self.active_requests) > 0

    def step(self) -> List[Request]:
        """Run one iteration of the continuous batching loop.

        This method does three things:
          1. PREFILL  — pull pending requests into empty slots (up to max_slots).
          2. DECODE   — one batched forward pass producing 1 new token per slot.
          3. EVICT    — remove completed requests and splice their KV slices out.

        Returns:
            A list of Request objects that completed during this step.
        """
        completed_this_step: List[Request] = []

        # ==================================================================
        # PHASE 1 — PREFILL: fill empty slots from the pending queue
        # ==================================================================
        # For each new request we run a *separate* prefill forward pass
        # (the full prompt), generate its first token, and splice the
        # resulting KV cache into the shared batch cache.
        #
        # Why separate prefills?  Each prompt has a different length.
        # Running them individually avoids expensive cross-sequence padding
        # during the computationally heavy prefill phase.  The resulting
        # per-layer KV tensors are then padded/concatenated along the batch
        # dimension to join the existing active batch.
        # ==================================================================
        while len(self.active_requests) < self.max_slots and self.pending_requests:
            req = self.pending_requests.pop(0)
            req.status = "active"

            with torch.no_grad():
                metrics.start_timer("token_generation_time")

                # --- Prefill forward pass ---
                # Run the full prompt through the model to populate its
                # KV cache and obtain logits for the *last* prompt token.
                prefill_mask = torch.ones(
                    (1, req.input_ids.shape[1]),
                    dtype=torch.long,
                    device=self.model.device,
                )
                outputs = self.model(
                    input_ids=req.input_ids,
                    attention_mask=prefill_mask,
                    use_cache=True,
                )

                # --- First token generation ---
                # Sample from the logits at the last position to produce
                # the first generated token for this request.
                logits = outputs.logits[:, -1, :]
                next_token = sample_next_token(logits, temperature=0.8, top_p=0.95)
                req.generated_tokens.append(next_token.item())

                new_cache = outputs.past_key_values

                # The attention mask for this request must cover all positions
                # the KV cache knows about: the original prompt tokens plus
                # the one token we just generated.
                new_mask = torch.ones(
                    (1, req.input_ids.shape[1] + 1),
                    dtype=torch.long,
                    device=self.model.device,
                )

                duration = metrics.stop_timer("token_generation_time")
                metrics.observe("time_to_first_token_ms", duration * 1000)

                # --- Splice into the shared batch KV cache ---
                if self.past_key_values is None:
                    # First active request — just adopt its cache directly.
                    self.past_key_values = new_cache
                    self.attention_mask = new_mask
                    self.active_requests.append(req)
                else:
                    # There are already active requests.  We need to align
                    # the sequence-length dimension (dim=2 in the KV tensors)
                    # before we can concatenate along the batch dimension (dim=0).
                    #
                    # KV tensor shape: [batch, heads, seq_len, head_dim]
                    #
                    # If the new request's prompt is shorter than the current
                    # batch's max seq_len, we left-pad its KV cache with zeros.
                    # If it's longer, we left-pad the existing batch instead.
                    # The attention_mask tracks which positions are real (1)
                    # vs padding (0), so the model ignores the padded slots.
                    existing_seq_len = self.past_key_values.layers[0].keys.shape[2]
                    new_seq_len = new_cache.layers[0].keys.shape[2]
                    diff = existing_seq_len - new_seq_len

                    if diff > 0:
                        # New request is shorter — pad it to match the batch.
                        # F.pad on dim=2 (seq_len): pad_spec (0, 0, diff, 0)
                        # means 0 padding on head_dim, `diff` on the left of seq_len.
                        for i in range(len(new_cache.layers)):
                            new_cache.layers[i].keys = F.pad(
                                new_cache.layers[i].keys, (0, 0, diff, 0)
                            )
                            new_cache.layers[i].values = F.pad(
                                new_cache.layers[i].values, (0, 0, diff, 0)
                            )
                        # Mask: left-pad with 0 so model ignores those positions.
                        new_mask = F.pad(new_mask, (diff, 0), value=0)

                    elif diff < 0:
                        # New request is longer — pad the *existing* batch.
                        diff = -diff
                        for i in range(len(self.past_key_values.layers)):
                            self.past_key_values.layers[i].keys = F.pad(
                                self.past_key_values.layers[i].keys, (0, 0, diff, 0)
                            )
                            self.past_key_values.layers[i].values = F.pad(
                                self.past_key_values.layers[i].values, (0, 0, diff, 0)
                            )
                        self.attention_mask = F.pad(
                            self.attention_mask, (diff, 0), value=0
                        )

                    # Concatenate along the batch dimension (dim=0).
                    for i in range(len(self.past_key_values.layers)):
                        self.past_key_values.layers[i].keys = torch.cat(
                            [
                                self.past_key_values.layers[i].keys,
                                new_cache.layers[i].keys,
                            ],
                            dim=0,
                        )
                        self.past_key_values.layers[i].values = torch.cat(
                            [
                                self.past_key_values.layers[i].values,
                                new_cache.layers[i].values,
                            ],
                            dim=0,
                        )

                    self.attention_mask = torch.cat(
                        [self.attention_mask, new_mask], dim=0
                    )
                    self.active_requests.append(req)

        # If no requests are active (nothing pending either), we're done.
        if not self.active_requests:
            return completed_this_step

        metrics.observe("batch_size", len(self.active_requests))

        # ==================================================================
        # PHASE 2 — DECODE: batched forward pass for one new token per slot
        # ==================================================================
        # We feed only the *last generated token* for each active request
        # (shape [batch, 1]).  The KV cache already contains all prior
        # context, so we don't need to re-process the full sequence.
        # This is what makes autoregressive decoding O(1) per token
        # instead of O(n).
        # ==================================================================
        current_input_ids = torch.tensor(
            [[req.generated_tokens[-1]] for req in self.active_requests],
            dtype=torch.long,
            device=self.model.device,
        )

        with torch.no_grad():
            metrics.start_timer("token_generation_time")

            # Single batched forward pass across all active slots.
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=self.attention_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
            )

            # Update the shared KV cache with the new key/value entries
            # the model just computed for this decode step.
            self.past_key_values = outputs.past_key_values

            # Sample the next token for each slot in the batch.
            logits = outputs.logits[:, -1, :]
            next_tokens = sample_next_token(logits, temperature=0.8, top_p=0.95)

            # Extend the attention mask by 1 column (the token we just fed in).
            # Every active slot gets a 1 because the token is real, not padding.
            self.attention_mask = torch.cat(
                [
                    self.attention_mask,
                    torch.ones(
                        (len(self.active_requests), 1),
                        dtype=torch.long,
                        device=self.model.device,
                    ),
                ],
                dim=1,
            )

            duration = metrics.stop_timer("token_generation_time")
            metrics.observe("time_per_output_token_ms", duration * 1000)
            metrics.inc("total_generated_tokens", len(self.active_requests))

            # ==============================================================
            # PHASE 3 — EVICT: identify and remove completed requests
            # ==============================================================
            # A request is complete when it has generated max_new_tokens.
            # We collect indices of finished slots, then surgically remove
            # their rows from the KV cache and attention mask tensors so
            # the remaining active requests can continue uninterrupted.
            # ==============================================================
            indices_to_remove = []
            for i, req in enumerate(self.active_requests):
                req.generated_tokens.append(next_tokens[i, 0].item())
                if len(req.generated_tokens) >= req.max_new_tokens:
                    req.status = "completed"
                    completed_this_step.append(req)
                    indices_to_remove.append(i)

            if indices_to_remove:
                keep_indices = [
                    i
                    for i in range(len(self.active_requests))
                    if i not in indices_to_remove
                ]
                self.active_requests = [
                    self.active_requests[i] for i in keep_indices
                ]

                if len(keep_indices) == 0:
                    # All slots finished — reset cache entirely.
                    self.past_key_values = None
                    self.attention_mask = None
                else:
                    # Slice out only the rows (batch indices) we're keeping.
                    keep_tensor = torch.tensor(
                        keep_indices, device=self.model.device
                    )
                    self.attention_mask = self.attention_mask[keep_tensor]
                    for layer in self.past_key_values.layers:
                        layer.keys = layer.keys[keep_tensor]
                        layer.values = layer.values[keep_tensor]

        # ==================================================================
        # PHASE 4 — METRICS: measure current KV cache memory footprint
        # ==================================================================
        # This lets us verify that memory stays bounded by max_slots.
        # ==================================================================
        if self.past_key_values is not None:
            cache_size_bytes = 0
            for layer in self.past_key_values.layers:
                cache_size_bytes += layer.keys.element_size() * layer.keys.nelement()
                cache_size_bytes += (
                    layer.values.element_size() * layer.values.nelement()
                )
            metrics.observe("kv_cache_size_mb", cache_size_bytes / (1024 * 1024))

        return completed_this_step


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
