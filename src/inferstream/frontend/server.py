import os
import grpc
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2
from inferstream.grpc.generated.inferstream.v1 import coordinator_pb2_grpc

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.channel = grpc.aio.insecure_channel("localhost:50051")
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

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return f.read()

@app.post("/api/generate")
async def submit_generate(req: GenerateRequest, request: Request):
    stub = request.app.state.stub
    
    length_enum = coordinator_pb2.SHORT
    if req.length.lower() == "medium":
        length_enum = coordinator_pb2.MEDIUM
    elif req.length.lower() == "long":
        length_enum = coordinator_pb2.LONG
        
    grpc_req = coordinator_pb2.SubmitRequestReq(
        parameters=coordinator_pb2.InferenceParameters(
            prompt=req.prompt,
            length=length_enum,
        )
    )
    res = await stub.SubmitRequest(grpc_req)
    return {"request_id": res.request_id}

@app.get("/api/result/{request_id}")
async def get_result(request_id: str, request: Request):
    stub = request.app.state.stub
    grpc_req = coordinator_pb2.GetResultReq(request_id=request_id)
    res = await stub.GetResult(grpc_req)
    
    status_str = "UNKNOWN"
    if res.status == coordinator_pb2.PENDING:
        status_str = "PENDING"
    elif res.status == coordinator_pb2.COMPLETED:
        status_str = "COMPLETED"
    elif res.status == coordinator_pb2.FAILED:
        status_str = "FAILED"
        
    return {
        "status": status_str,
        "generated_text": res.generated_text
    }

def main():
    import uvicorn
    uvicorn.run("inferstream.frontend.server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
