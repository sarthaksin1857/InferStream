import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferstream.inference.engine import ContinuousBatchingEngine, sample_next_token

def main():
    print("🚀 Initializing model and tokenizer...")
    model_name = "distilgpt2" 
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    print("\n🧠 Setting up ContinuousBatchingEngine...")
    engine = ContinuousBatchingEngine(model, tokenizer, max_slots=2, max_seq_len=64)
    
    prompt1 = "What is your favorite color"
    prompt2 = "What is the capital of France"
    
    # We add the first prompt
    print(f"\n📥 Adding Request 1: '{prompt1}'")
    engine.add_request(prompt1, max_new_tokens=10)
    
    # We add the second prompt
    print(f"📥 Adding Request 2: '{prompt2}'")
    engine.add_request(prompt2, max_new_tokens=10)

    print("\n==================================================")
    print("🔍 STARTING INTERACTIVE DEBUG LOOP")
    print("==================================================\n")

    step_num = 1
    while engine.has_pending_or_active():
        print(f"\n--- 🔄 Engine Step {step_num} ---")
        
        # ------------------------------------------------------------------
        # We will manually peek into what the engine is about to do!
        # ------------------------------------------------------------------
        
        # Look at pending requests that are about to be prefilled
        num_free_slots = sum(1 for s in engine.slots if s.is_free)
        requests_to_prefill = engine.pending_requests[:num_free_slots]
        
        if requests_to_prefill:
            print("🚀 PHASE: PREFILL")
            for i, req in enumerate(requests_to_prefill):
                seq_len = req.input_ids.shape[1]
                print(f"   -> Prefilling Request '{req.prompt}'")
                print(f"   -> Input shape: {req.input_ids.shape}")
                print(f"   -> Attention Mask (Prefill) will be shape: (1, {seq_len}) - all 1s")
                
        # Look at active decode slots
        decode_slots = [s for s in engine.slots if not s.is_free and not s.needs_prefill]
        if decode_slots:
            print("🧠 PHASE: DECODE (Batched)")
            num_active = len(decode_slots)
            max_batch_seq_len = max(engine.global_cache.slot_seq_lens[s.slot_id] for s in decode_slots) + 1
            
            print(f"   -> Active Decode Slots: {[s.slot_id for s in decode_slots]}")
            print(f"   -> Max Batch Seq Len: {max_batch_seq_len}")
            
            # Reconstruct what the engine will build for the attention mask to show you
            batched_attention_mask = torch.zeros((num_active, max_batch_seq_len), dtype=torch.long)
            position_ids = torch.zeros((num_active, 1), dtype=torch.long)
            
            for i, s in enumerate(decode_slots):
                s_len = engine.global_cache.slot_seq_lens[s.slot_id]
                batched_attention_mask[i, :s_len + 1] = 1
                position_ids[i, 0] = s_len
                print(f"      Slot {s.slot_id} | Cache Len: {s_len} | Pos ID: {s_len}")
            
            print(f"\n   -> Batched Attention Mask Shape: {batched_attention_mask.shape}")
            print(f"   -> Batched Attention Mask Tensor:\n{batched_attention_mask}")
            print(f"\n   -> Position IDs Tensor:\n{position_ids}")

        # ------------------------------------------------------------------
        # Actually execute the engine step
        # ------------------------------------------------------------------
        completed = engine.step()
        
        # Print generated tokens this step
        for s in engine.slots:
            if not s.is_free and s.request.generated_tokens:
                latest_token = s.request.generated_tokens[-1]
                decoded_word = tokenizer.decode([latest_token])
                print(f"   ✨ Slot {s.slot_id} Generated Token: '{decoded_word}' (ID: {latest_token})")
        
        for req in completed:
            text = tokenizer.decode(req.input_ids[0].tolist() + req.generated_tokens)
            print(f"\n✅ Request {req.request_id} FINISHED:")
            print(f"   {text}\n")

        step_num += 1

if __name__ == "__main__":
    main()
