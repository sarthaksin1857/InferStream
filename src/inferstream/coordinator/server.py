import asyncio
import logging
import uuid
import grpc

from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2
from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from inferstream.coordinator.state_store import StateStore

class CoordinatorServiceServicer(coordinator_pb2_grpc.CoordinatorServiceServicer):
    def __init__(self):
        super().__init__()
        self.state_store = StateStore()

    async def SubmitRequest(
        self,
        request: coordinator_pb2.SubmitRequestReq,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.SubmitRequestRes:
        request_id = str(uuid.uuid4())
        logger.info(f"Received SubmitRequest: prompt='{request.parameters.prompt}' (id={request_id})")

        task = coordinator_pb2.InferenceTask(request_id=request_id, parameters=request.parameters)
        await self.state_store.register_job(request_id, task)
        
        return coordinator_pb2.SubmitRequestRes(
            request_id=request_id,
        )

    async def GetResult(
        self,
        request: coordinator_pb2.GetResultReq,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.GetResultRes:
        status = self.state_store.get_status(request.request_id)

        if status is None:
            logger.info(
                f"GetResult: request_id={request.request_id} → FAILED (unknown id)"
            )
            return coordinator_pb2.GetResultRes(
                status=coordinator_pb2.FAILED,
                generated_text="",
            )

        # Map the proto enum to a readable name for the log
        status_names = {
            coordinator_pb2.PENDING:   "PENDING",
            coordinator_pb2.ASSIGNED:  "ASSIGNED",
            coordinator_pb2.COMPLETED: "COMPLETED",
            coordinator_pb2.FAILED:    "FAILED",
        }
        logger.info(
            f"GetResult: request_id={request.request_id} "
            f"→ {status_names.get(status, status)}"
        )

        res = coordinator_pb2.GetResultRes(status=status)
        if status == coordinator_pb2.COMPLETED:
            res.generated_text = (
                self.state_store.get_completed_results(request.request_id) or ""
            )

        return res

    async def RegisterWorker(
        self,
        request: coordinator_pb2.RegisterWorkerReq,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.RegisterWorkerRes:
        logger.info(f"Received RegisterWorker: worker_id={request.worker.worker_id}")
        self.state_store.register_worker(request.worker.worker_id)
        return coordinator_pb2.RegisterWorkerRes(success=True)

    async def GetWork(
        self, request: coordinator_pb2.GetWorkReq, context: grpc.aio.ServicerContext
    ) -> coordinator_pb2.GetWorkRes:
        # Honour the worker's requested batch size (capped at MAX_SLOTS = 8).
        # If the worker doesn't set the field it defaults to 0, so we fall
        # back to 8 so old clients still get a full batch.
        max_batch = request.max_batch_size if request.max_batch_size > 0 else 8
        logger.info(
            f"Received GetWork: worker_id={request.worker_id} max_batch={max_batch}"
        )

        worker = coordinator_pb2.Worker(worker_id=request.worker_id)
        assigned_tasks = await self.state_store.assignWork(worker, max_batch_size=max_batch)

        return coordinator_pb2.GetWorkRes(tasks=assigned_tasks)

    async def SubmitResults(
        self,
        request: coordinator_pb2.SubmitResultsReq,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.SubmitResultsRes:
        logger.info(f"Received SubmitResults: worker_id={request.worker_id}, results={len(request.results)}")
        for result in request.results:
            self.state_store.mark_completed(result.request_id, result.generated_text)
        return coordinator_pb2.SubmitResultsRes(success=True)


async def serve() -> None:
    server = grpc.aio.server()
    coordinator_pb2_grpc.add_CoordinatorServiceServicer_to_server(
        CoordinatorServiceServicer(), server
    )
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting Coordinator server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


def main():
    asyncio.run(serve())

if __name__ == "__main__":
    main()
