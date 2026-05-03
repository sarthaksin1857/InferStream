import asyncio
import logging
import uuid
import grpc

from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2
from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinatorServiceServicer(coordinator_pb2_grpc.CoordinatorServiceServicer):
    def __init__(self):
        super().__init__()
        # Store request state for polling: {request_id: {"status": coordinator_pb2.PENDING, "text": ""}}
        self.requests_state = {}
        self.workers = set()

    async def SubmitRequest(
        self,
        request: coordinator_pb2.SubmitRequestReq,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.SubmitRequestRes:
        request_id = str(uuid.uuid4())
        logger.info(f"Received SubmitRequest: prompt='{request.prompt}' (id={request_id})")
        
        # Store in state
        self.requests_state[request_id] = {
            "status": coordinator_pb2.PENDING,
            "text": ""
        }
        
        return coordinator_pb2.SubmitRequestRes(
            request_id=request_id,
        )

    async def GetResult(
        self,
        request: coordinator_pb2.GetResultReq,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.GetResultRes:
        logger.info(f"Received GetResult: request_id={request.request_id}")
        state = self.requests_state.get(request.request_id)

        return coordinator_pb2.GetResultRes(
            status=coordinator_pb2.COMPLETED,
            generated_text="lol"
        )
        
        if not state:
            # If not found, returning FAILED for now
            return coordinator_pb2.GetResultRes(
                status=coordinator_pb2.FAILED,
                generated_text=""
            )
            
        return coordinator_pb2.GetResultRes(
            status=state["status"],
            generated_text=state["text"]
        )

    async def RegisterWorker(
        self,
        request: coordinator_pb2.RegisterWorkerReq,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.RegisterWorkerRes:
        logger.info(f"Received RegisterWorker: worker_id={request.worker_id}")
        self.workers.add(request.worker_id)
        return coordinator_pb2.RegisterWorkerRes(success=True)

    async def GetWork(
        self, request: coordinator_pb2.GetWorkReq, context: grpc.aio.ServicerContext
    ) -> coordinator_pb2.GetWorkRes:
        logger.info(f"Received GetWork: worker_id={request.worker_id}")
        return coordinator_pb2.GetWorkRes(tasks=[])

    async def SubmitResults(
        self,
        request: coordinator_pb2.SubmitResultsReq,
        context: grpc.aio.ServicerContext,
    ) -> coordinator_pb2.SubmitResultsRes:
        logger.info(f"Received SubmitResults: worker_id={request.worker_id}, results={len(request.results)}")
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
