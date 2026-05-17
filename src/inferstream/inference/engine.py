import os
from typing import Any, Dict, List, Optional, Tuple
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
from transformers.cache_utils import Cache

from inferstream.metrics import metrics

# ---------------------------------------------------------------------------
# Request: the unit of work flowing through the engine
# ---------------------------------------------------------------------------
@dataclass
class Request:
    """A single inference request managed by the ContinuousBatchingEngine."""
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
@dataclass
class Slot:
    """A fixed inference slot that can hold one active request at a time."""
    slot_id: int
    request: Optional[Request] = None
    needs_prefill: bool = False

    @property
    def is_free(self) -> bool:
        return self.request is None

    @property
    def is_active(self) -> bool:
        return self.request is not None and not self.needs_prefill

    def clear(self) -> None:
        """Fully reset the slot."""
        self.request = None
        self.needs_prefill = False


from transformers.cache_utils import DynamicCache

# ---------------------------------------------------------------------------
# StaticSlotCache
# ---------------------------------------------------------------------------
class StaticSlotCache(DynamicCache):
    """
    A custom Cache that pre-allocates KV memory for a fixed number of slots
    and allows batched decoding across a subset of active slots without padding
    the storage tensor.
    """
    def __init__(self, config, max_slots: int, max_seq_len: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.num_hidden_layers = config.num_hidden_layers
        
        self.num_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 0))
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // getattr(config, "num_attention_heads", 1)

        # Pre-allocate global cache tensors
        self.k_cache = torch.zeros(
            (self.num_hidden_layers, max_slots, self.num_heads, max_seq_len, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (self.num_hidden_layers, max_slots, self.num_heads, max_seq_len, self.head_dim),
            dtype=dtype,
            device=device,
        )
        
        # Routing state set before each forward pass
        self.active_slots: List[int] = []
        self.slot_seq_lens: Dict[int, int] = {i: 0 for i in range(max_slots)}
        
        # Dummy layers removed.
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the global static cache for the currently active slots.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        max_batch_seq_len = 0
        
        for i, slot_id in enumerate(self.active_slots):
            current_len = self.slot_seq_lens[slot_id]
            # Write new states directly into the pre-allocated cache
            self.k_cache[layer_idx, slot_id, :, current_len : current_len + seq_len, :] = key_states[i]
            self.v_cache[layer_idx, slot_id, :, current_len : current_len + seq_len, :] = value_states[i]
            
            max_batch_seq_len = max(max_batch_seq_len, current_len + seq_len)

        # Return the rectangular slice for the current batch up to the max length.
        # This memory is gathered for the forward pass, but the cache itself remains static.
        k_out = self.k_cache[layer_idx, self.active_slots, :, :max_batch_seq_len, :]
        v_out = self.v_cache[layer_idx, self.active_slots, :, :max_batch_seq_len, :]

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if not self.active_slots:
            return 0
        return max(self.slot_seq_lens[s] for s in self.active_slots)



# ---------------------------------------------------------------------------
# ContinuousBatchingEngine
# ---------------------------------------------------------------------------
class ContinuousBatchingEngine:
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

        self.pending_requests: List[Request] = []
        self.slots: List[Slot] = [Slot(slot_id=i) for i in range(max_slots)]
        self.next_request_id = 0
        
        # Initialize the global static cache
        self.global_cache = StaticSlotCache(
            config=self.model.config,
            max_slots=max_slots,
            max_seq_len=max_seq_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )

    @property
    def active_requests(self) -> List[Request]:
        return [s.request for s in self.slots if s.request is not None]

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        external_id: Optional[str] = None,
    ) -> int:
        req_id = self.next_request_id
        self.next_request_id += 1

        if getattr(self.tokenizer, "chat_template", None) is not None:
            # Format as a conversation if a chat template is available
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_dict=True,
                return_tensors="pt"
            ).input_ids.to(self.model.device)
        else:
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
        return len(self.pending_requests) > 0 or any(
            not s.is_free for s in self.slots
        )

    def step(self) -> List[Request]:
        completed_this_step: List[Request] = []

        # ==================================================================
        # PHASE 1 — FILL
        # ==================================================================
        for slot in self.slots:
            if slot.is_free and self.pending_requests:
                req = self.pending_requests.pop(0)
                req.status = "active"
                slot.request = req
                slot.needs_prefill = True
                # Reset cache length for this slot
                self.global_cache.slot_seq_lens[slot.slot_id] = 0

        active_count = sum(1 for s in self.slots if not s.is_free)
        if active_count == 0:
            return completed_this_step

        metrics.observe("batch_size", active_count)

        # ==================================================================
        # PHASE 2 — PREFILL
        # ==================================================================
        prefilled_this_step: set = set()

        for slot in self.slots:
            if not slot.needs_prefill:
                continue

            req = slot.request
            with torch.no_grad():
                metrics.start_timer("token_generation_time")

                seq_len = req.input_ids.shape[1]
                prefill_mask = torch.ones(
                    (1, seq_len), dtype=torch.long, device=self.model.device
                )
                
                # We can construct explicit position ids to be safe
                position_ids = torch.arange(seq_len, dtype=torch.long, device=self.model.device).unsqueeze(0)

                # Route cache updates to this specific slot
                self.global_cache.active_slots = [slot.slot_id]

                outputs = self.model(
                    input_ids=req.input_ids,
                    attention_mask=prefill_mask,
                    position_ids=position_ids,
                    past_key_values=self.global_cache,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                next_token = sample_next_token(logits, temperature=0.8, top_p=0.95)
                req.generated_tokens.append(next_token.item())

                # Update the sequence length for this slot
                self.global_cache.slot_seq_lens[slot.slot_id] += seq_len

                slot.needs_prefill = False
                prefilled_this_step.add(slot.slot_id)

                duration = metrics.stop_timer("token_generation_time")
                metrics.observe("time_to_first_token_ms", duration * 1000)

        # ==================================================================
        # PHASE 3 — DECODE (BATCHED)
        # ==================================================================
        active_decode_slots = [
            slot for slot in self.slots
            if not slot.is_free and not slot.needs_prefill and slot.slot_id not in prefilled_this_step
        ]

        if active_decode_slots:
            with torch.no_grad():
                metrics.start_timer("token_generation_time")
                num_active = len(active_decode_slots)

                # Decode phase: each slot gets exactly 1 token
                batched_input_ids = torch.tensor(
                    [[s.request.generated_tokens[-1]] for s in active_decode_slots],
                    dtype=torch.long,
                    device=self.model.device,
                )

                # Route cache updates
                self.global_cache.active_slots = [s.slot_id for s in active_decode_slots]

                # Maximum sequence length among active slots (AFTER adding this new token)
                max_batch_seq_len = max(self.global_cache.slot_seq_lens[s.slot_id] for s in active_decode_slots) + 1
                
                # Build batched attention mask and position IDs
                batched_attention_mask = torch.zeros(
                    (num_active, max_batch_seq_len), 
                    dtype=torch.long, 
                    device=self.model.device
                )
                position_ids = torch.zeros(
                    (num_active, 1), 
                    dtype=torch.long, 
                    device=self.model.device
                )

                for i, s in enumerate(active_decode_slots):
                    # Length of valid context before this new token
                    s_len = self.global_cache.slot_seq_lens[s.slot_id]
                    # The new token will bring the length to s_len + 1
                    batched_attention_mask[i, :s_len + 1] = 1
                    # Explicitly provide the position ID for the new token
                    position_ids[i, 0] = s_len

                outputs = self.model(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                    position_ids=position_ids,
                    past_key_values=self.global_cache,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                for i, s in enumerate(active_decode_slots):
                    req = s.request
                    next_token = sample_next_token(logits[i:i+1], temperature=0.8, top_p=0.95)
                    req.generated_tokens.append(next_token.item())
                    # Increment length for the appended token
                    self.global_cache.slot_seq_lens[s.slot_id] += 1

                duration = metrics.stop_timer("token_generation_time")
                metrics.observe("time_per_output_token_ms", duration * 1000 / num_active)
                metrics.inc("total_generated_tokens", num_active)

        # ==================================================================
        # PHASE 4 — EVICT
        # ==================================================================
        for slot in self.slots:
            if slot.is_free:
                continue

            req = slot.request
            seq_len = self.global_cache.slot_seq_lens[slot.slot_id]
            seq_len_exceeded = seq_len >= self.max_seq_len

            if len(req.generated_tokens) >= req.max_new_tokens or seq_len_exceeded:
                req.status = "completed"
                completed_this_step.append(req)
                slot.clear()
                self.global_cache.slot_seq_lens[slot.slot_id] = 0

        # ==================================================================
        # PHASE 5 — METRICS
        # ==================================================================
        # In a statically allocated cache, the memory footprint is constant!
        cache_size_bytes = (
            self.global_cache.k_cache.element_size() * self.global_cache.k_cache.nelement() +
            self.global_cache.v_cache.element_size() * self.global_cache.v_cache.nelement()
        )
        if cache_size_bytes > 0:
            metrics.observe("kv_cache_size_mb", cache_size_bytes / (1024 * 1024))

        return completed_this_step


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9
) -> torch.Tensor:
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
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
