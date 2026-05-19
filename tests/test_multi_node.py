import asyncio
import uuid
import pytest
from inferstream.coordinator.state_store import StateStore
from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2

def test_multi_node_assignment():
    asyncio.run(_test_multi_node_assignment())

async def _test_multi_node_assignment():
    store = StateStore()

    # Register two workers
    worker1 = coordinator_pb2.Worker(worker_id=str(uuid.uuid4()), model_name="test-model", maxMemoryAssignment=1024)
    worker2 = coordinator_pb2.Worker(worker_id=str(uuid.uuid4()), model_name="test-model", maxMemoryAssignment=1024)
    
    store.register_worker(worker1)
    store.register_worker(worker2)

    # Submit 20 tasks
    task_ids = []
    for i in range(20):
        req_id = f"task-{i}"
        task_ids.append(req_id)
        task = coordinator_pb2.InferenceTask(
            request_id=req_id,
            parameters=coordinator_pb2.InferenceParameters(prompt=f"Prompt {i}", length=coordinator_pb2.SHORT)
        )
        await store.register_job(req_id, task)

    assert store.pending_queue.qsize() == 20

    # Worker 1 requests 10 tasks
    assigned1 = await store.assignWork(worker1, max_batch_size=10)
    assert len(assigned1) == 10

    # Worker 2 requests 10 tasks
    assigned2 = await store.assignWork(worker2, max_batch_size=10)
    assert len(assigned2) == 10

    # Ensure no overlap
    assigned1_ids = {t.request_id for t in assigned1}
    assigned2_ids = {t.request_id for t in assigned2}
    
    overlap = assigned1_ids.intersection(assigned2_ids)
    assert len(overlap) == 0, f"Overlap found in task assignment: {overlap}"

    # Verify queue is empty
    assert store.pending_queue.empty()

    # Simulate worker 1 completing its tasks
    for t in assigned1:
        store.mark_completed(t.request_id, f"Result for {t.request_id}")
        
    for t in assigned1:
        assert store.is_ready(t.request_id)
        assert store.get_completed_results(t.request_id) == f"Result for {t.request_id}"

    # Verify no dead workers pruned prematurely
    await store.prune_dead_workers(timeout_sec=10.0)
    assert len(store.get_workers()) == 2
