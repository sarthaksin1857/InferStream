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
# Slot: a fixed inference slot that holds one request at a time
# ---------------------------------------------------------------------------
# Each slot owns its own KV cache and attention mask.  When the request
# finishes, the slot is fully cleared — KV cache is discarded, and the
# next request starts from scratch at its own prompt length.  This prevents
# the unbounded sequence-length growth that occurs with a shared padded
# batch cache.
# ---------------------------------------------------------------------------

@dataclass
class Slot:
    """A fixed inference slot that can hold one active request at a time.

    Fields:
        slot_id:        Immutable index of this slot (0 .. max_slots-1).
        request:        The currently assigned Request, or None if free.
        kv_cache:       Per-slot KV cache (DynamicCache), shape per layer:
                        [1, heads, seq_len, head_dim].  None when free.
        attention_mask: Per-slot attention mask, shape [1, seq_len].
                        None when free.
        needs_prefill:  True if the slot was just filled and needs a full
                        prefill forward pass before it can decode.
    """
    slot_id: int
    request: Optional[Request] = None
    kv_cache: object = None            # DynamicCache — typed as object to
    attention_mask: Optional[torch.Tensor] = None  # avoid import dependency
    needs_prefill: bool = False

    @property
    def is_free(self) -> bool:
        return self.request is None

    @property
    def is_active(self) -> bool:
        return self.request is not None and not self.needs_prefill

    def clear(self) -> None:
        """Fully reset the slot, discarding KV cache and request."""
        self.request = None
        self.kv_cache = None
        self.attention_mask = None
        self.needs_prefill = False


# ---------------------------------------------------------------------------
# ContinuousBatchingEngine
# ---------------------------------------------------------------------------
# This is the core of the inference server.  It operates with a fixed number
# of Slots, each holding at most one request at a time.
#
# Unlike the previous shared-batch design, each slot maintains its own
# independent KV cache.  This means:
#
#   • A new request entering a vacated slot starts at its own prompt length —
#     it does NOT inherit the sequence length of surviving requests.
#   • Memory is bounded by: max_slots × max_seq_len × per_token_kv_bytes.
#   • No cross-slot padding is needed.
#
# The trade-off is that each slot runs its own forward pass (batch=1)
# instead of one large batched forward pass.  For small max_slots on a
# single device this is acceptable — extreme padding in the batched
# approach often negates the GPU parallelism benefit anyway.
#
# Lifecycle of a request through the engine:
#
#   add_request()          →  enqueued in self.pending_requests
#   step() PHASE 1 — FILL →  assigned to a free Slot, needs_prefill = True
#   step() PHASE 2 — PREFILL → full prompt forward pass, first token
#                              generated, KV cache stored in the Slot
#   step() PHASE 3 — DECODE  → single-token forward pass, token appended
#   step() PHASE 4 — EVICT   → slot fully cleared, request returned
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """Slot-based continuous batching engine with per-request KV caches.

    Args:
        model:       A HuggingFace causal LM already on the target device.
        tokenizer:   The corresponding tokenizer.
        max_slots:   Maximum number of requests generating tokens at once.
                     This directly caps KV-cache memory usage.
        max_seq_len: Hard cap on per-slot sequence length (prompt + generated).
                     Requests whose total would exceed this are still accepted
                     but will be stopped early.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_slots: int = 8,
        max_seq_len: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len

        # Queues
        self.pending_requests: List[Request] = []

        # Fixed-size slot array — the core of the new design
        self.slots: List[Slot] = [Slot(slot_id=i) for i in range(max_slots)]

        # Auto-incrementing request id counter
        self.next_request_id = 0

    # ------------------------------------------------------------------
    # Convenience accessors (backwards-compatible surface for tests/worker)
    # ------------------------------------------------------------------

    @property
    def active_requests(self) -> List[Request]:
        """List of all requests currently assigned to a slot (active or prefilling)."""
        return [s.request for s in self.slots if s.request is not None]

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
        but no forward pass happens until step() pulls it into a free slot.

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
        return len(self.pending_requests) > 0 or any(
            not s.is_free for s in self.slots
        )

    def step(self) -> List[Request]:
        """Run one iteration of the continuous batching loop.

        This method does four things:
          1. FILL    — assign pending requests to free slots.
          2. PREFILL — run full-prompt forward passes for newly filled slots.
          3. DECODE  — run single-token forward passes for all active slots.
          4. EVICT   — clear completed slots and return finished requests.

        Returns:
            A list of Request objects that completed during this step.
        """
        completed_this_step: List[Request] = []

        # ==================================================================
        # PHASE 1 — FILL: assign pending requests to free slots
        # ==================================================================
        # Walk the slot array and fill any empty slot with the next pending
        # request.  The request is marked "active" and the slot is flagged
        # needs_prefill so PHASE 2 knows to run a full forward pass.
        # ==================================================================
        for slot in self.slots:
            if slot.is_free and self.pending_requests:
                req = self.pending_requests.pop(0)
                req.status = "active"
                slot.request = req
                slot.needs_prefill = True

        # Count active slots for metrics
        active_count = sum(1 for s in self.slots if not s.is_free)
        if active_count == 0:
            return completed_this_step

        metrics.observe("batch_size", active_count)

        # ==================================================================
        # PHASE 2 — PREFILL: full-prompt forward pass for newly filled slots
        # ==================================================================
        # Each newly filled slot runs its own forward pass over the full
        # prompt to populate its KV cache and generate its first token.
        #
        # The KV cache is stored directly in the Slot — shape per layer:
        #   [1, heads, prompt_len, head_dim]
        #
        # No padding, no concatenation with other slots.
        # ==================================================================

        # Track which slots were prefilled THIS step so PHASE 3 can skip them
        # (they already produced their first token here; decoding next step).
        prefilled_this_step: set = set()

        for slot in self.slots:
            if not slot.needs_prefill:
                continue

            req = slot.request
            with torch.no_grad():
                metrics.start_timer("token_generation_time")

                # --- Prefill forward pass ---
                # Run the full prompt through the model to populate the
                # slot's KV cache and obtain logits for the last prompt token.
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
                logits = outputs.logits[:, -1, :]
                next_token = sample_next_token(logits, temperature=0.8, top_p=0.95)
                req.generated_tokens.append(next_token.item())

                # Store KV cache and mask in the slot — this is the slot's
                # own private cache, shape [1, heads, prompt_len, head_dim].
                slot.kv_cache = outputs.past_key_values
                slot.attention_mask = torch.ones(
                    (1, req.input_ids.shape[1] + 1),  # prompt + first token
                    dtype=torch.long,
                    device=self.model.device,
                )
                slot.needs_prefill = False
                prefilled_this_step.add(slot.slot_id)

                duration = metrics.stop_timer("token_generation_time")
                metrics.observe("time_to_first_token_ms", duration * 1000)

        # ==================================================================
        # PHASE 3 — DECODE: single-token forward pass for each active slot
        # ==================================================================
        # Each active slot (has a KV cache, not just prefilled) runs its own
        # forward pass feeding only the last generated token.  The slot's
        # private KV cache provides all prior context.
        #
        # We iterate over slots individually — no cross-slot padding needed.
        # ==================================================================
        for slot in self.slots:
            if slot.is_free or slot.needs_prefill:
                continue

            # Skip slots that were just prefilled this step — they already
            # generated their first token in PHASE 2.
            if slot.slot_id in prefilled_this_step:
                continue

            req = slot.request
            with torch.no_grad():
                metrics.start_timer("token_generation_time")

                # Feed only the last generated token — the KV cache has
                # all prior context.
                input_id = torch.tensor(
                    [[req.generated_tokens[-1]]],
                    dtype=torch.long,
                    device=self.model.device,
                )

                outputs = self.model(
                    input_ids=input_id,
                    attention_mask=slot.attention_mask,
                    past_key_values=slot.kv_cache,
                    use_cache=True,
                )

                # Update the slot's KV cache with the new entries
                slot.kv_cache = outputs.past_key_values

                # Sample the next token
                logits = outputs.logits[:, -1, :]
                next_token = sample_next_token(logits, temperature=0.8, top_p=0.95)
                req.generated_tokens.append(next_token.item())

                # Extend the slot's attention mask by 1 column
                slot.attention_mask = torch.cat(
                    [
                        slot.attention_mask,
                        torch.ones(
                            (1, 1),
                            dtype=torch.long,
                            device=self.model.device,
                        ),
                    ],
                    dim=1,
                )

                duration = metrics.stop_timer("token_generation_time")
                metrics.observe("time_per_output_token_ms", duration * 1000)
                metrics.inc("total_generated_tokens", 1)

        # ==================================================================
        # PHASE 4 — EVICT: clear completed slots
        # ==================================================================
        # A request is complete when it has generated max_new_tokens, or
        # when its sequence length would exceed max_seq_len.
        #
        # The slot is fully cleared — KV cache discarded, request removed.
        # The next pending request will start fresh in this slot on the
        # next call to step().
        # ==================================================================
        for slot in self.slots:
            if slot.is_free:
                continue

            req = slot.request
            seq_len_exceeded = (
                slot.attention_mask is not None
                and slot.attention_mask.shape[1] >= self.max_seq_len
            )

            if len(req.generated_tokens) >= req.max_new_tokens or seq_len_exceeded:
                req.status = "completed"
                completed_this_step.append(req)
                # FULL RESET — this is the key fix.
                # The slot's KV cache is discarded entirely.  The next
                # request will start from its own prompt length.
                slot.clear()

        # ==================================================================
        # PHASE 5 — METRICS: measure current KV cache memory footprint
        # ==================================================================
        # Sum across all active slots to get total KV cache memory.
        # This should stay bounded by max_slots × max_seq_len.
        # ==================================================================
        cache_size_bytes = 0
        for slot in self.slots:
            if slot.kv_cache is not None:
                for layer in slot.kv_cache.layers:
                    cache_size_bytes += layer.keys.element_size() * layer.keys.nelement()
                    cache_size_bytes += (
                        layer.values.element_size() * layer.values.nelement()
                    )
        if cache_size_bytes > 0:
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
