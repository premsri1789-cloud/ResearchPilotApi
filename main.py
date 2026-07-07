import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#custom modules
from app.graph.Ingestion import setup_collection, ingest_pipeline
from app.graph.Langgraph_engine import run_agent

# Initialize FastAPI app
app = FastAPI(title="Research Copilot API", description="Backend for Multimodal CRAG System")

# Configure CORS so Angular (usually running on port 4200) can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, change this to ["http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
COLLECTION_NAME = "Research_documents"
UPLOAD_DIR = "/tmp/temp_uploads" # CHANGED TO /tmp FOR SERVERLESS

# Ensure upload and DB directories exist on startup
os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs("./app/db", exist_ok=True)
setup_collection(COLLECTION_NAME)

# --- Pydantic Models for Request/Response Validation ---
class ChatRequest(BaseModel):
    session_id: str
    user_id: str      # Added user_id
    question: str
    history: list     # Accept history directly from Angular

class ChatResponse(BaseModel):
    answer: str

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"status": "online", "message": "Research Copilot API is running."}

@app.post("/api/upload")
async def upload_document(
    session_id: str = Form(...), 
    user_id: str = Form(...),  # Added user_id to form data
    file: UploadFile = File(...)
):
    """Receives a PDF and triggers the ingestion pipeline."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"File saved. Starting ingestion for User: {user_id}, Session: {session_id}...")
        
        # Pass user_id to the ingestion pipeline
        ingest_pipeline(file_path=file_path, session_id=session_id, user_id=user_id, COLLECTIONNAME=COLLECTION_NAME)
        
        return {"status": "success", "message": "Document processed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Executes the LangGraph pipeline."""
    try:
        # Pass the history provided by Angular directly into the agent
        answer = run_agent(
            session_id=request.session_id, 
            user_id=request.user_id,
            question=request.question,
            history=request.history # Pass history
        )
        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"Error during graph execution: {e}")
        raise HTTPException(status_code=500, detail="An error occurred.")