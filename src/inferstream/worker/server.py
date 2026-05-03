import asyncio
import logging
import uuid
import grpc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2
from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2_grpc
from inferstream.inference.engine import generate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map TaskLength enum to max_new_tokens
LENGTH_MAPPING = {
    coordinator_pb2.LENGTH_UNSPECIFIED: 50,
    coordinator_pb2.SHORT: 256,
    coordinator_pb2.MEDIUM: 1024,
    coordinator_pb2.LONG: 2048,
}

async def run_worker():
    # 1. Setup device and load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Worker starting up. Using device: {device}")

    model_name = "distilgpt2"
    logger.info(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    # 2. Connect to Coordinator
    worker_id = str(uuid.uuid4())
    coordinator_addr = "localhost:50051"
    
    # We use an async channel
    async with grpc.aio.insecure_channel(coordinator_addr) as channel:
        stub = coordinator_pb2_grpc.CoordinatorServiceStub(channel)
        
        # 3. Register Worker
        logger.info(f"Registering worker {worker_id} with coordinator...")
        register_req = coordinator_pb2.RegisterWorkerReq(
            worker=coordinator_pb2.Worker(
                worker_id=worker_id,
                maxMemoryAssignment=8000 # Dummy value for now
            )
        )
        res = await stub.RegisterWorker(register_req)
        if not res.success:
            logger.error("Failed to register worker with coordinator.")
            return
            
        logger.info("Successfully registered. Starting polling loop...")

        # 4. Polling Loop
        while True:
            try:
                # Poll for work
                work_req = coordinator_pb2.GetWorkReq(
                    worker_id=worker_id,
                    max_batch_size=1
                )
                work_res = await stub.GetWork(work_req)

                if not work_res.tasks:
                    # No work, sleep and poll again
                    await asyncio.sleep(1.0)
                    continue

                # Process tasks
                logger.info(f"Received {len(work_res.tasks)} task(s) from coordinator.")
                
                results = []
                for task in work_res.tasks:
                    logger.info(f"Processing task {task.request_id}...")
                    
                    prompt = task.parameters.prompt
                    length_enum = task.parameters.length
                    max_new_tokens = LENGTH_MAPPING.get(length_enum, 50)
                    
                    # Tokenize
                    inputs = tokenizer(
                        [prompt],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)

                    # Generate
                    output_strings = generate(
                        model=model,
                        tokenizer=tokenizer,
                        inputs=dict(inputs),
                        max_new_tokens=max_new_tokens,
                    )
                    
                    # Package result
                    # We passed a single prompt, so we take the first output string
                    generated_text = output_strings[0] if output_strings else ""
                    
                    res_item = coordinator_pb2.InferenceResult(
                        request_id=task.request_id,
                        generated_text=generated_text
                    )
                    results.append(res_item)
                    
                    logger.info(f"Task {task.request_id} completed. Generated {len(generated_text)} chars.")

                # Submit results back
                submit_req = coordinator_pb2.SubmitResultsReq(
                    worker_id=worker_id,
                    results=results
                )
                await stub.SubmitResults(submit_req)
                logger.info("Results successfully submitted to coordinator.")

            except grpc.RpcError as e:
                logger.error(f"gRPC Error during polling: {e.details()}")
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(2.0)

def main():
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker shutting down.")

if __name__ == "__main__":
    main()
