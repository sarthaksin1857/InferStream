import asyncio
import logging
import uuid
import grpc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2
from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2_grpc
from inferstream.inference.engine import ContinuousBatchingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map TaskLength enum to max_new_tokens
LENGTH_MAPPING = {
    coordinator_pb2.LENGTH_UNSPECIFIED: 50,
    coordinator_pb2.SHORT: 256,
    coordinator_pb2.MEDIUM: 1024,
    coordinator_pb2.LONG: 2048,
}

# Maximum number of concurrent generation slots.
# This directly caps the KV-cache memory this worker will use.
MAX_SLOTS = 8

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

    # Create the continuous batching engine — this is the core inference
    # loop that manages KV cache slots.  We keep it alive across polling
    # iterations so in-flight requests continue generating tokens while
    # we fetch more work.
    engine = ContinuousBatchingEngine(
        model=model, tokenizer=tokenizer, max_slots=MAX_SLOTS
    )

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

        # Mapping from engine request_id → coordinator request_id so we
        # can correctly report results back to the coordinator.
        # (The engine uses integer ids internally; the coordinator uses
        # string UUIDs.)
        engine_to_coordinator_id: dict[int, str] = {}

        # 4. Polling Loop
        while True:
            try:
                # -------------------------------------------------------
                # 4a. Poll for new work to fill empty slots
                # -------------------------------------------------------
                # We ask the coordinator for up to MAX_SLOTS tasks at once.
                # The engine will accept them and start generating on the
                # next step() call.
                available_slots = engine.max_slots - (
                    len(engine.active_requests) + len(engine.pending_requests)
                )

                if available_slots > 0:
                    work_req = coordinator_pb2.GetWorkReq(
                        worker_id=worker_id,
                        max_batch_size=available_slots,
                    )
                    work_res = await stub.GetWork(work_req)

                    if work_res.tasks:
                        logger.info(
                            f"Received {len(work_res.tasks)} task(s) from coordinator."
                        )

                    for task in work_res.tasks:
                        prompt = task.parameters.prompt
                        length_enum = task.parameters.length
                        max_new_tokens = LENGTH_MAPPING.get(length_enum, 50)

                        # Feed the task into the engine.  We pass the
                        # coordinator's request_id as external_id so we
                        # can map completed requests back later.
                        engine_id = engine.add_request(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            external_id=task.request_id,
                        )
                        engine_to_coordinator_id[engine_id] = task.request_id
                        logger.info(
                            f"Enqueued task {task.request_id} as engine slot {engine_id}"
                        )

                # -------------------------------------------------------
                # 4b. Run one step of continuous batching
                # -------------------------------------------------------
                # Each step() call generates exactly one token for every
                # active slot and returns any requests that just finished.
                if engine.has_pending_or_active():
                    completed = engine.step()

                    # -------------------------------------------------------
                    # 4c. Submit completed results back to the coordinator
                    # -------------------------------------------------------
                    if completed:
                        results = []
                        for req in completed:
                            coordinator_request_id = (
                                req.external_id
                                or engine_to_coordinator_id.pop(req.request_id, "")
                            )
                            generated_text = tokenizer.decode(
                                req.generated_tokens, skip_special_tokens=True
                            )

                            res_item = coordinator_pb2.InferenceResult(
                                request_id=coordinator_request_id,
                                generated_text=generated_text,
                            )
                            results.append(res_item)

                            logger.info(
                                f"Task {coordinator_request_id} completed. "
                                f"Generated {len(generated_text)} chars."
                            )

                        submit_req = coordinator_pb2.SubmitResultsReq(
                            worker_id=worker_id, results=results
                        )
                        await stub.SubmitResults(submit_req)
                        logger.info("Results successfully submitted to coordinator.")
                else:
                    # Nothing in flight — sleep briefly before polling again
                    # to avoid busy-spinning.
                    await asyncio.sleep(1.0)

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
