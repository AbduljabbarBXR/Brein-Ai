from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from orchestrator import Orchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Brein AI", description="Modular brain-inspired AI system")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
memory_manager = MemoryManager(db_path="memory/brein_memory.db")
orchestrator = Orchestrator(memory_manager)

# Pydantic models for API requests
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class IngestRequest(BaseModel):
    content: str
    content_type: str = "stable"

@app.get("/api/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "message": "Brein AI orchestrator is running"}

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    """Handle user queries with memory augmentation"""
    return await orchestrator.process_query(request.query, request.session_id)

@app.post("/api/ingest")
async def ingest_content(request: IngestRequest):
    """Ingest new content into memory"""
    return await orchestrator.ingest_content(request.content, request.content_type)

@app.get("/api/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    return memory_manager.get_memory_stats()

@app.get("/api/memory/search")
async def search_memory(query: str, top_k: int = 5):
    """Search memory for similar content"""
    return {"results": memory_manager.search_memory(query, top_k)}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Brein AI", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)