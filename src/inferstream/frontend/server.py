import os
import grpc
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2
from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2_grpc

@asynccontextmanager
async def lifespan(app: FastAPI):
    coordinator_addr = os.environ.get("COORDINATOR_ADDR", "localhost:50051")
    app.state.channel = grpc.aio.insecure_channel(coordinator_addr)
    app.state.stub = coordinator_pb2_grpc.CoordinatorServiceStub(app.state.channel)
    yield
    await app.state.channel.close()

app = FastAPI(title="InferStream Frontend", lifespan=lifespan)

# Mount static files for JS/CSS if needed
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class GenerateRequest(BaseModel):
    prompt: str
    length: str = "short"
    model_name: str = ""

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return f.read()

@app.post("/api/generate")
async def submit_generate(req: GenerateRequest, request: Request):
    stub = request.app.state.stub
    
    # Check for active workers
    status_req = coordinator_pb2.GetSystemStatusReq()
    status_res = await stub.GetSystemStatus(status_req)
    if not status_res.workers:
        raise HTTPException(status_code=503, detail="No active workers available to process the request.")
    
    length_enum = coordinator_pb2.SHORT
    if req.length.lower() == "medium":
        length_enum = coordinator_pb2.MEDIUM
    elif req.length.lower() == "long":
        length_enum = coordinator_pb2.LONG
        
    grpc_req = coordinator_pb2.SubmitRequestReq(
        parameters=coordinator_pb2.InferenceParameters(
            prompt=req.prompt,
            length=length_enum,
            model_name=req.model_name
        )
    )
    res = await stub.SubmitRequest(grpc_req)
    return {"request_id": res.request_id}

@app.get("/api/result/{request_id}")
async def get_result(request_id: str, request: Request):
    stub = request.app.state.stub
    grpc_req = coordinator_pb2.GetResultReq(request_id=request_id)
    res = await stub.GetResult(grpc_req)

    status_map = {
        coordinator_pb2.PENDING:   "PENDING",
        coordinator_pb2.ASSIGNED:  "ASSIGNED",
        coordinator_pb2.COMPLETED: "COMPLETED",
        coordinator_pb2.FAILED:    "FAILED",
    }
    status_str = status_map.get(res.status, "UNKNOWN")

    return {
        "status": status_str,
        "generated_text": res.generated_text,
    }

@app.get("/api/status")
async def get_status(request: Request):
    stub = request.app.state.stub
    grpc_req = coordinator_pb2.GetSystemStatusReq()
    res = await stub.GetSystemStatus(grpc_req)
    
    workers = []
    for w in res.workers:
        workers.append({
            "worker_id": w.worker_id,
            "model_name": w.model_name
        })
        
    return {"workers": workers}

def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="InferStream Frontend Server")
    parser.add_argument("--coordinator", type=str, default="localhost:50051",
                        help="gRPC address of the coordinator (default: localhost:50051)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port to listen on")
    args = parser.parse_args()
    os.environ.setdefault("COORDINATOR_ADDR", args.coordinator)
    uvicorn.run("inferstream.frontend.server:app", host="0.0.0.0", port=args.port, reload=True)

if __name__ == "__main__":
    main()
