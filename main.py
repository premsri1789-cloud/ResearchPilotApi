import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.graph.Ingestion import setup_collection, ingest_pipeline
from app.graph.Langgraph_engine import run_agent

app = FastAPI()

# Configure local Angular CORS accessibility explicitly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "Research_documents"

@app.on_event("startup")
def on_startup():
    setup_collection(COLLECTION_NAME)

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...), 
    session_id: str = Form(...), 
    user_id: str = Form(...)
):
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        ingest_pipeline(file_path, session_id, user_id, COLLECTION_NAME)
        return {"message": "Document ingested successfully inside local vector store."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/chat")
async def chat_endpoint(data: dict):
    try:
        session_id = data.get("session_id", "default_session")
        user_id = data.get("user_id", "default_user")
        question = data.get("question", "")
        history = data.get("history", [])
        
        answer = run_agent(session_id, user_id, question, history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))