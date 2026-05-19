import argparse
import asyncio
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

def send_post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e)}

def send_get(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e)}

async def run_single_request(frontend_url, prompt, length, model_name, executor, poll_interval=1.0):
    start_time = time.time()
    
    # 1. Submit
    data = {"prompt": prompt, "length": length, "model_name": model_name}
    loop = asyncio.get_running_loop()
    submit_res = await loop.run_in_executor(executor, send_post, f"{frontend_url}/api/generate", data)
    
    if "error" in submit_res:
        return {"error": submit_res["error"], "time": time.time() - start_time}
    if "request_id" not in submit_res:
        return {"error": f"Invalid response: {submit_res}", "time": time.time() - start_time}
        
    request_id = submit_res["request_id"]
    
    # 2. Poll
    while True:
        await asyncio.sleep(poll_interval)
        poll_res = await loop.run_in_executor(executor, send_get, f"{frontend_url}/api/result/{request_id}")
        
        if "error" in poll_res:
            return {"error": poll_res["error"], "time": time.time() - start_time}
            
        status = poll_res.get("status")
        if status == "COMPLETED":
            end_time = time.time()
            return {
                "status": "COMPLETED",
                "text": poll_res.get("generated_text", ""),
                "time": end_time - start_time
            }
        elif status == "FAILED":
            return {"error": "Request failed on server", "time": time.time() - start_time}

async def main():
    parser = argparse.ArgumentParser(description="Load Test InferStream")
    parser.add_argument("--url", default="http://localhost:8000", help="Frontend URL")
    parser.add_argument("--requests", type=int, default=10, help="Number of concurrent requests to send")
    parser.add_argument("--length", choices=["short", "medium", "long"], default="medium", help="Generation length")
    parser.add_argument("--model", default="", help="Model name (optional)")
    parser.add_argument("--prompt", default="Explain the theory of relativity in detail.", help="Prompt to use")
    parser.add_argument("--count-tokens", action="store_true", help="Attempt to count actual tokens using transformers (requires transformers installed)")
    args = parser.parse_args()

    print(f"Starting load test...")
    print(f"URL: {args.url}")
    print(f"Requests: {args.requests}")
    print(f"Length: {args.length}")
    
    executor = ThreadPoolExecutor(max_workers=min(200, args.requests + 10))
    
    start_time = time.time()
    
    tasks = []
    for _ in range(args.requests):
        tasks.append(run_single_request(args.url, args.prompt, args.length, args.model, executor))
        
    print(f"Sent {args.requests} requests concurrently. Waiting for responses...")
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    success_count = sum(1 for r in results if r.get("status") == "COMPLETED")
    error_count = len(results) - success_count
    
    print(f"\n--- Results ---")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful requests: {success_count} / {args.requests}")
    if error_count > 0:
        print(f"Errors: {error_count}")
        # Only print first few errors
        errors_printed = 0
        for r in results:
            if "error" in r:
                print(f"  - {r['error']}")
                errors_printed += 1
                if errors_printed >= 5:
                    print(f"  - ... and {error_count - 5} more errors")
                    break
                
    if success_count > 0:
        avg_time = sum(r["time"] for r in results if r.get("status") == "COMPLETED") / success_count
        print(f"Average request latency: {avg_time:.2f} seconds")
        
        # Estimate tokens
        length_mapping = {"short": 50, "medium": 150, "long": 300}
        estimated_tokens_per_request = length_mapping.get(args.length, 50)
        
        total_tokens = 0
        if args.count_tokens:
            try:
                from transformers import AutoTokenizer
                model_to_load = args.model if args.model else "meta-llama/Llama-3.1-8B-Instruct" # Default fallback
                print(f"Loading tokenizer for exact token count...")
                tokenizer = AutoTokenizer.from_pretrained(model_to_load)
                for r in results:
                    if r.get("status") == "COMPLETED":
                        total_tokens += len(tokenizer.encode(r["text"]))
                print(f"Exact tokens generated: {total_tokens}")
            except Exception as e:
                print(f"Failed to count exact tokens ({e}). Using estimates.")
                total_tokens = success_count * estimated_tokens_per_request
        else:
            total_tokens = success_count * estimated_tokens_per_request
            print(f"Estimated tokens generated: ~{total_tokens} (based on {args.length} length)")
            
        tok_per_sec = total_tokens / total_time
        print(f"Throughput: {tok_per_sec:.2f} tokens/second")

if __name__ == "__main__":
    asyncio.run(main())
