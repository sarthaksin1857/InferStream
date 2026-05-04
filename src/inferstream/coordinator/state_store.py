import asyncio
import logging
import uuid
import grpc

from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2
from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2_grpc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



import asyncio
from typing import Any, Optional, Dict, Set
from collections import deque

# Holds information about workers, tasks, statuses
# All work is keyed by request_id
class StateStore:
    def __init__(self):
        # Maps request_id -> RequestStatus
        self.statuses: Dict[str, int] = {}
        # Maps request_id -> InferenceTask (or request context)
        self.tasks: Dict[str, Any] = {}
        # Maps request_id -> generated string result
        self.results: Dict[str, str] = {}
        # Single queue for all pending requests
        self.pending_queue: asyncio.Queue = asyncio.Queue()
        # Set of registered worker IDs
        self.workers: Set[str] = set()

    def register_worker(self, worker_id: str) -> None:
        """Registers a worker with the state store."""
        self.workers.add(worker_id)
        logger.info(f"Registered worker {worker_id} in state store")

    async def register_job(self, request_id: str, task_data: Any) -> None:
        """Adds a request_id into all our states, puts it in the queue."""
        self.statuses[request_id] = coordinator_pb2.PENDING
        self.tasks[request_id] = task_data
        
        await self.pending_queue.put(request_id)
        logger.info(f"Registered job {request_id}")

    def get_status(self, request_id: str) -> Optional[int]:
        """Returns the status of a request_id."""
        return self.statuses.get(request_id)

    def mark_completed(self, request_id: str, result: str) -> Any:
        """Marks request as completed, stores result, and cleans up active tracking.
        Returns the original task data as requested.
        """
        if request_id in self.statuses:
            self.statuses[request_id] = coordinator_pb2.COMPLETED
            self.results[request_id] = result
            logger.info(f"Job {request_id} completed.")
            
            # Mark as done in the queue only when actually finished
            try:
                self.pending_queue.task_done()
            except ValueError:
                pass # In case task_done was called more times than put
                
            return self.tasks.get(request_id)
        return None

    def is_ready(self, request_id: str) -> bool:
        """Checks if a request is completed."""
        return self.statuses.get(request_id) == coordinator_pb2.COMPLETED

    def get_completed_results(self, request_id: str) -> Optional[str]:
        """Returns the completed results."""
        if self.is_ready(request_id):
            return self.results.get(request_id)
        return None

    async def assignWork(self, worker: coordinator_pb2.Worker, max_batch_size: int = 8) -> list:
        """Pops up to max_batch_size items from the pending queue and assigns
        them to the requesting worker.

        Args:
            worker:         The worker descriptor (currently used for logging).
            max_batch_size: Maximum number of tasks to hand out in one call.
                            The worker passes its number of free engine slots
                            so we never send more work than it can handle.
        """
        assigned_tasks = []
        for _ in range(max_batch_size):
            try:
                request_id = self.pending_queue.get_nowait()
                self.statuses[request_id] = coordinator_pb2.ASSIGNED
                assigned_tasks.append(self.tasks.get(request_id))
            except asyncio.QueueEmpty:
                break  # No more work available right now

        if assigned_tasks:
            logger.info(
                f"Assigned {len(assigned_tasks)} task(s) to worker {worker.worker_id}"
            )
        return assigned_tasks
