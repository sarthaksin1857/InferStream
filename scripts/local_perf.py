import argparse
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferstream.metrics import metrics
from inferstream.inference.engine import ContinuousBatchingEngine

def run_local_perf(model_name: str, batch_size: int, num_requests: int, max_new_tokens: int):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    print(f"Initializing engine with {batch_size} slots...")
    # NOTE: Set max_seq_len accordingly so we have enough KV cache capacity
    engine = ContinuousBatchingEngine(model=model, tokenizer=tokenizer, max_slots=batch_size, max_seq_len=1024)

    print(f"Adding {num_requests} requests...")
    prompt = "Explain the theory of relativity in detail."
    for _ in range(num_requests):
        engine.add_request(prompt, max_new_tokens=max_new_tokens)

    print("Starting generation...")
    start_time = time.time()
    
    completed_requests = []
    step_count = 0
    
    while engine.has_pending_or_active():
        completed = engine.step()
        completed_requests.extend(completed)
        step_count += 1
        
        # Simple progress logger
        if step_count % 20 == 0:
            active_count = len(engine.active_requests)
            print(f"Generating tokens... Active slots: {active_count}/{batch_size} | Completed: {len(completed_requests)}/{num_requests}")

    end_time = time.time()
    total_time = end_time - start_time
    
    total_tokens = sum(len(req.generated_tokens) for req in completed_requests)
    tok_per_sec = total_tokens / total_time
    
    print("\n=============================================")
    print("           LOCAL PERFORMANCE RESULTS         ")
    print("=============================================")
    print(f"Model:                  {model_name}")
    print(f"Batch Size (slots):     {batch_size}")
    print(f"Total Requests:         {num_requests}")
    print(f"Max New Tokens/Req:     {max_new_tokens}")
    print("---------------------------------------------")
    print(f"Total Time:             {total_time:.2f} s")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Engine Throughput:      {tok_per_sec:.2f} tokens/sec")
    print("=============================================\n")
    
    # Print the internal engine metrics
    metrics.print_summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Inference Engine Performance Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--batch-size", type=int, default=40, help="Max slots for the continuous batching engine")
    parser.add_argument("--requests", type=int, default=40, help="Number of concurrent requests to queue")
    parser.add_argument("--tokens", type=int, default=50, help="Max new tokens to generate per request")
    args = parser.parse_args()
    
    run_local_perf(args.model, args.batch_size, args.requests, args.tokens)
