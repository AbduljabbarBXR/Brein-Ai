from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager

from memory_manager import MemoryManager
from orchestrator import Orchestrator
from chat_manager import ChatManager
# from model_exporter import ModelExporter  # Temporarily disabled - requires onnxruntime
from sync_manager import SyncManager
from web_fetcher import WebFetcher
from audit_logger import AuditLogger
from test_harness import TestHarness
from profiler import PerformanceProfiler
from vision_processor import VisionProcessor
from audio_processor import AudioProcessor
from multi_modal_fusion import MultiModalFusionEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global orchestrator_initialized
    logger.info("Starting Brein AI server...")

    try:
        # Initialize orchestrator with SAL connections
        await orchestrator.initialize()
        orchestrator_initialized = True
        logger.info("Orchestrator initialized successfully with SAL integration")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        # Continue with degraded functionality

    yield

    # Shutdown
    logger.info("Shutting down Brein AI server...")

app = FastAPI(title="Brein AI", description="Modular brain-inspired AI system with mobile sync", lifespan=lifespan)

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
chat_manager = ChatManager(db_path="chats/brein_chats.db")
orchestrator = Orchestrator(memory_manager, chat_manager)

# Initialize multi-modal components
try:
    vision_processor = VisionProcessor()
    logger.info("Vision processor initialized successfully")
except Exception as e:
    logger.warning(f"Vision processor initialization failed: {e}")
    vision_processor = None

try:
    audio_processor = AudioProcessor()
    logger.info("Audio processor initialized successfully")
except Exception as e:
    logger.warning(f"Audio processor initialization failed (audio dependencies may be missing): {e}")
    logger.info("Continuing without audio processing capabilities")
    audio_processor = None

try:
    fusion_engine = MultiModalFusionEngine(
        memory_manager=memory_manager,
        concept_extractor=orchestrator.concept_extractor,
        vision_processor=vision_processor,
        audio_processor=audio_processor
    )
    logger.info("Multi-modal fusion engine initialized successfully")
except Exception as e:
    logger.warning(f"Multi-modal fusion engine initialization failed: {e}")
    fusion_engine = None

# model_exporter = ModelExporter()  # Temporarily disabled - requires onnxruntime
sync_manager = SyncManager()
web_fetcher = WebFetcher(memory_manager)
audit_logger = AuditLogger()
test_harness = TestHarness(memory_manager, orchestrator)
performance_profiler = PerformanceProfiler()

# Initialize orchestrator with SAL connections
orchestrator_initialized = False

# Pydantic models for API requests
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    enable_web_access: bool = False

class IngestRequest(BaseModel):
    content: str
    content_type: str = "stable"

class DeviceRegisterRequest(BaseModel):
    device_id: str
    device_name: str
    device_type: str = "mobile"
    capabilities: List[str] = []

class SyncDeltaRequest(BaseModel):
    device_id: str
    since_version: int = 0

class WebFetchRequest(BaseModel):
    url: str
    enable_web_access: bool = False

class QuarantineReviewRequest(BaseModel):
    quarantine_path: str
    approved: bool
    reviewer_notes: str = ""

@app.get("/api/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "message": "Brein AI orchestrator is running"}

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    """Handle user queries with memory augmentation and optional web access"""
    import asyncio

    # Audit log the query operation
    audit_logger.log_operation(
        operation_type="user_query",
        user_id="user",  # In production, get from auth
        session_id=request.session_id,
        details={
            "query_length": len(request.query),
            "web_access_requested": request.enable_web_access
        }
    )

    try:
        # Add 30-second timeout to prevent server hangs
        result = await asyncio.wait_for(
            orchestrator.process_query(request.query, request.session_id, request.enable_web_access),
            timeout=30.0
        )

        # Log memory operations if any
        if result.get("memory_chunks"):
            audit_logger.log_memory_operation(
                operation="retrieve",
                node_ids=[chunk["node_id"] for chunk in result["memory_chunks"]],
                user_id="user",
                session_id=request.session_id
            )

        return result

    except asyncio.TimeoutError:
        logger.error(f"Query timeout for session {request.session_id}: {request.query[:100]}...")
        raise HTTPException(status_code=504, detail="Query processing timed out. Please try again.")

@app.post("/api/ingest")
async def ingest_content(request: IngestRequest):
    """Ingest new content into memory"""
    return await orchestrator.ingest_content(request.content, request.content_type)

@app.get("/api/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    return memory_manager.get_memory_stats()

@app.get("/api/neural-mesh/full")
async def get_full_neural_mesh():
    """Get complete neural mesh data for visualization"""
    try:
        # Get the neural mesh data directly from the loaded instance
        neural_mesh = memory_manager.neural_mesh

        # Convert edges dict to serializable format for frontend
        edges_serializable = {}
        for (node_a, node_b), edge_data in neural_mesh.edges.items():
            edge_key = f"{node_a}::{node_b}"
            edges_serializable[edge_key] = edge_data

        # Return structured data for frontend visualization
        return {
            "nodes": neural_mesh.nodes,
            "edges": edges_serializable,
            "stats": neural_mesh.get_mesh_stats()
        }
    except Exception as e:
        logger.error(f"Error loading neural mesh data: {e}")
        return {"error": "Failed to load neural mesh data"}

@app.get("/api/memory/search")
async def search_memory(query: str, top_k: int = 5):
    """Search memory for similar content"""
    return {"results": memory_manager.search_memory(query, top_k)}

# Concept analysis endpoints
@app.get("/api/concepts/search")
async def search_concepts(query: str, top_k: int = 10):
    """Search concepts by semantic similarity"""
    results = orchestrator.concept_extractor.search_concepts_by_similarity(query, top_k)
    return {"results": [{"concept_id": cid, "similarity": sim} for cid, sim in results]}

@app.get("/api/concepts/{concept_id}")
async def get_concept_details(concept_id: str):
    """Get detailed information about a specific concept"""
    if concept_id not in orchestrator.concept_extractor.concepts:
        raise HTTPException(status_code=404, detail="Concept not found")

    concept = orchestrator.concept_extractor.concepts[concept_id]
    evolution = orchestrator.concept_extractor.get_concept_evolution(concept_id)
    related = orchestrator.concept_extractor.find_related_concepts(concept_id, top_k=10)
    context = orchestrator.concept_extractor.get_concept_context(concept_id, max_contexts=5)

    return {
        "concept": concept,
        "evolution": evolution,
        "related_concepts": [{"concept_id": cid, "weight": w} for cid, w in related],
        "context_history": context
    }

@app.get("/api/concepts/{concept_id}/related")
async def get_related_concepts(concept_id: str, top_k: int = 10):
    """Get concepts related to a specific concept"""
    if concept_id not in orchestrator.concept_extractor.concept_graph:
        raise HTTPException(status_code=404, detail="Concept not found")

    related = orchestrator.concept_extractor.find_related_concepts(concept_id, top_k)
    return {"related_concepts": [{"concept_id": cid, "weight": w} for cid, w in related]}

@app.get("/api/concepts/{concept_id}/evolution")
async def get_concept_evolution(concept_id: str):
    """Get temporal evolution data for a concept"""
    if concept_id not in orchestrator.concept_extractor.concepts:
        raise HTTPException(status_code=404, detail="Concept not found")

    evolution = orchestrator.concept_extractor.get_concept_evolution(concept_id)
    return {"evolution": evolution}

@app.get("/api/concepts/stats")
async def get_concept_stats():
    """Get comprehensive concept system statistics"""
    stats = orchestrator.concept_extractor.get_concept_stats()
    return {"stats": stats}

@app.post("/api/concepts/build-ontology")
async def build_concept_ontology():
    """Build hierarchical concept relationships (ontology)"""
    orchestrator.concept_extractor.build_concept_ontology()
    return {"status": "completed", "message": "Concept ontology built successfully"}

@app.post("/api/concepts/update-relationships")
async def update_concept_relationships():
    """Update semantic relationships in the concept graph"""
    orchestrator.concept_extractor.update_semantic_relationships()
    return {"status": "completed", "message": "Concept relationships updated successfully"}

@app.post("/api/concepts/cleanup")
async def cleanup_old_concepts(days_threshold: int = 90):
    """Remove concepts that haven't been accessed recently"""
    removed_count = orchestrator.concept_extractor.cleanup_old_concepts(days_threshold)
    return {"status": "completed", "concepts_removed": removed_count}

@app.get("/api/concepts/graph")
async def get_concept_graph():
    """Get concept relationship graph for visualization"""
    # Convert NetworkX graph to serializable format
    nodes = []
    for node_id, node_data in orchestrator.concept_extractor.concept_graph.nodes(data=True):
        concept_name = orchestrator.concept_extractor.concepts.get(node_id, {}).get('name', node_id)
        importance = orchestrator.concept_extractor.concepts.get(node_id, {}).get('importance_score', 0.5)
        nodes.append({
            "id": node_id,
            "name": concept_name,
            "importance": importance,
            "group": 1  # For visualization grouping
        })

    edges = []
    for u, v, data in orchestrator.concept_extractor.concept_graph.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": data.get('weight', 0.1)
        })

    return {"nodes": nodes, "edges": edges}

@app.get("/api/concepts/hierarchy")
async def get_concept_hierarchy():
    """Get concept hierarchy for visualization"""
    # Convert NetworkX DiGraph to serializable format
    nodes = []
    for node_id in orchestrator.concept_extractor.concept_hierarchy.nodes():
        concept_name = orchestrator.concept_extractor.concepts.get(node_id, {}).get('name', node_id)
        nodes.append({
            "id": node_id,
            "name": concept_name
        })

    edges = []
    for u, v, data in orchestrator.concept_extractor.concept_hierarchy.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": data.get('weight', 0.1)
        })

    return {"nodes": nodes, "edges": edges}

@app.post("/api/memory/consolidate/decay")
async def apply_memory_decay(hours_since_last_decay: float = 24):
    """Apply time-based memory decay to all nodes"""
    result = memory_manager.consolidator.apply_memory_decay(hours_since_last_decay)
    return {"status": "completed", "result": result}

@app.post("/api/memory/consolidate/similar")
async def consolidate_similar_memories():
    """Find and consolidate similar memories to prevent redundancy"""
    result = memory_manager.consolidator.consolidate_similar_memories()
    return {"status": "completed", "result": result}

@app.get("/api/memory/consolidate/health")
async def get_memory_health_report():
    """Generate comprehensive memory health report"""
    report = memory_manager.consolidator.get_memory_health_report()
    return {"status": "success", "report": report}

@app.post("/api/memory/consolidate/reinforce")
async def reinforce_memory_nodes(node_ids: List[str], strength: float = 0.5, reason: str = "manual_reinforcement"):
    """Manually reinforce specific memory nodes"""
    result = memory_manager.consolidator.reinforce_memory(node_ids, strength, reason)
    return {"status": "completed", "result": result}

# Chat management endpoints
@app.post("/api/chats")
async def create_chat_session():
    """Create a new chat session."""
    session_id = chat_manager.create_chat_session()
    return {"session_id": session_id, "status": "created"}

@app.get("/api/chats")
async def list_chat_sessions():
    """Get all chat sessions."""
    sessions = chat_manager.get_chat_sessions()
    return {"chats": sessions}

@app.get("/api/chats/{session_id}")
async def get_chat_history(session_id: str):
    """Get the full message history for a chat session."""
    messages = chat_manager.get_chat_history(session_id)
    session_info = chat_manager.get_session_info(session_id)

    if not session_info:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "session": session_info,
        "messages": messages
    }

@app.delete("/api/chats/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages."""
    session_info = chat_manager.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Chat session not found")

    chat_manager.delete_chat_session(session_id)
    return {"status": "deleted", "session_id": session_id}

@app.put("/api/chats/{session_id}/title")
async def update_chat_title(session_id: str, title: str):
    """Update the title of a chat session."""
    session_info = chat_manager.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Chat session not found")

    chat_manager.update_chat_title(session_id, title)
    return {"status": "updated", "session_id": session_id, "title": title}

# Mobile/Sync endpoints
@app.post("/api/sync/register-device")
async def register_device(request: DeviceRegisterRequest):
    """Register a device for synchronization"""
    success = sync_manager.register_device(
        request.device_id,
        request.device_name,
        request.device_type,
        request.capabilities
    )
    return {"success": success, "device_id": request.device_id}

@app.post("/api/sync/delta")
async def get_sync_delta(request: SyncDeltaRequest):
    """Get synchronization delta for a device"""
    delta = sync_manager.create_sync_delta(request.device_id, request.since_version)
    return delta

@app.post("/api/sync/apply-delta")
async def apply_sync_delta(delta: Dict[str, Any]):
    """Apply synchronization delta from a device"""
    success = sync_manager.apply_sync_delta(delta)
    return {"success": success}

@app.get("/api/sync/status/{device_id}")
async def get_sync_status(device_id: str):
    """Get synchronization status for a device"""
    return sync_manager.get_sync_status(device_id)

@app.get("/api/sync/devices")
async def get_registered_devices():
    """Get list of all registered devices"""
    return {"devices": sync_manager.get_registered_devices()}

@app.post("/api/sync/create-bundle/{device_id}")
async def create_offline_bundle(device_id: str, max_nodes: int = 1000):
    """Create an offline bundle for a device"""
    bundle_path = sync_manager.create_offline_bundle(device_id, max_nodes)
    if bundle_path:
        return {"success": True, "bundle_path": bundle_path}
    else:
        raise HTTPException(status_code=500, detail="Failed to create offline bundle")

# Web fetch endpoints
@app.post("/api/web/fetch")
async def fetch_web_content(request: WebFetchRequest):
    """Fetch and process web content through safety pipeline"""
    if not request.enable_web_access:
        # Log security event for denied web access
        audit_logger.log_security_event(
            event_type="web_access_denied",
            severity="low",
            source="api_call",
            details={"url": request.url, "reason": "web_access_disabled"}
        )
        raise HTTPException(status_code=403, detail="Web access is disabled for this query")

    # Log web fetch operation
    audit_logger.log_operation(
        operation_type="web_fetch_request",
        user_id="user",
        details={"url": request.url}
    )

    result = web_fetcher.fetch_and_process_pipeline(request.url)

    # Log the fetch result
    audit_logger.log_web_fetch(
        url=request.url,
        success=result.get("success", False),
        safety_warnings=result.get("safety_warnings", []),
        quarantine_path=result.get("quarantine_path"),
        user_id="user"
    )

    return result

@app.post("/api/web/review")
async def review_quarantined_content(request: QuarantineReviewRequest):
    """Review and approve/reject quarantined content"""
    # Log content review operation
    audit_logger.log_operation(
        operation_type="content_review",
        user_id="reviewer",  # In production, get from auth
        details={
            "quarantine_path": request.quarantine_path,
            "approved": request.approved,
            "reviewer_notes": request.reviewer_notes
        }
    )

    success = web_fetcher.approve_quarantined_content(
        request.quarantine_path,
        request.approved,
        request.reviewer_notes
    )

    # Log the review result
    audit_logger.log_content_review(
        quarantine_path=request.quarantine_path,
        approved=request.approved,
        reviewer_id="reviewer",
        notes=request.reviewer_notes
    )

    return {"success": success}

@app.get("/api/web/quarantine/stats")
async def get_quarantine_stats():
    """Get quarantine statistics"""
    return web_fetcher.get_quarantine_stats()

@app.get("/api/web/quarantine/list")
async def list_quarantined_content():
    """List all quarantined content for review"""
    import os
    quarantine_files = []

    if os.path.exists("quarantine"):
        for filename in os.listdir("quarantine"):
            if filename.endswith('.json'):
                filepath = os.path.join("quarantine", filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        quarantine_files.append({
                            "filename": filename,
                            "path": filepath,
                            "url": data["fetch_result"]["url"],
                            "title": data["fetch_result"]["title"],
                            "status": data.get("status", "pending_review"),
                            "quarantine_timestamp": data["quarantine_timestamp"],
                            "word_count": data["fetch_result"]["metadata"]["word_count"]
                        })
                except Exception as e:
                    logger.error(f"Error reading quarantine file {filename}: {e}")

    return {"quarantined_content": quarantine_files}

# Audit endpoints
@app.get("/api/audit/summary")
async def get_audit_summary(days: int = 7):
    """Get audit summary for the specified period"""
    return audit_logger.get_audit_summary(days)

@app.post("/api/audit/search")
async def search_audit_logs(query: Dict[str, Any], log_type: str = "all", limit: int = 100):
    """Search audit logs"""
    return {"results": audit_logger.search_logs(query, log_type, limit)}

# Testing and profiling endpoints
@app.post("/api/test/run-comprehensive")
async def run_comprehensive_test():
    """Run comprehensive test suite"""
    results = await test_harness.run_comprehensive_test()
    return results

@app.post("/api/test/benchmark")
async def run_performance_benchmark(iterations: int = 100):
    """Run performance benchmark"""
    results = await test_harness.run_performance_benchmark(iterations)
    return results

@app.post("/api/profiler/start")
async def start_performance_monitoring():
    """Start performance monitoring"""
    performance_profiler.start_monitoring()
    return {"status": "started"}

@app.post("/api/profiler/stop")
async def stop_performance_monitoring():
    """Stop performance monitoring"""
    performance_profiler.stop_monitoring()
    return {"status": "stopped"}

@app.get("/api/profiler/current")
async def get_current_performance_metrics():
    """Get current performance metrics"""
    return performance_profiler.get_current_metrics()

@app.get("/api/profiler/summary")
async def get_performance_summary(hours: int = 1):
    """Get performance summary for the last N hours"""
    return performance_profiler.get_performance_summary(hours)

@app.get("/api/profiler/operations")
async def get_operation_performance(operation: Optional[str] = None, hours: int = 1):
    """Get operation performance metrics"""
    return performance_profiler.get_operation_performance(operation, hours)

@app.get("/api/profiler/health")
async def get_system_health():
    """Get system health status"""
    return performance_profiler.get_health_status()

@app.post("/api/profiler/report")
async def generate_performance_report():
    """Generate and save performance report"""
    filepath = performance_profiler.save_performance_report()
    return {"report_path": filepath}

# Multi-modal processing endpoints
@app.post("/api/vision/analyze")
async def analyze_image(image_data: Dict[str, Any]):
    """
    Analyze an image and extract visual concepts.

    Expected input: {"image": "base64_string_or_path", "context": {...}}
    """
    if not vision_processor:
        raise HTTPException(status_code=503, detail="Vision processing not available")

    try:
        result = vision_processor.process_image(
            image_data.get("image"),
            context=image_data.get("context")
        )
        return result
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/api/audio/analyze")
async def analyze_audio(audio_data: Dict[str, Any]):
    """
    Analyze audio and extract speech and audio concepts.

    Expected input: {"audio": "base64_string_or_path", "context": {...}}
    """
    if not audio_processor:
        raise HTTPException(status_code=503, detail="Audio processing not available")

    try:
        result = audio_processor.process_audio(
            audio_data.get("audio"),
            context=audio_data.get("context")
        )
        return result
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

@app.post("/api/audio/record")
async def record_audio(duration: int = 5, save_path: Optional[str] = None):
    """
    Record audio from microphone.

    Args:
        duration: Recording duration in seconds
        save_path: Optional path to save the recording
    """
    if not audio_processor:
        raise HTTPException(status_code=503, detail="Audio processing not available")

    try:
        audio_segment = audio_processor.record_audio(duration, save_path)
        if audio_segment:
            return {
                "success": True,
                "duration": len(audio_segment) / 1000,
                "message": f"Recorded {len(audio_segment) / 1000:.1f} seconds of audio"
            }
        else:
            raise HTTPException(status_code=500, detail="Audio recording failed")
    except Exception as e:
        logger.error(f"Audio recording failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio recording failed: {str(e)}")

@app.post("/api/multimodal/fuse")
async def fuse_multi_modal_inputs(inputs: Dict[str, Any]):
    """
    Process and fuse multiple modalities of input.

    Expected input: {"text": "string", "image": "base64_or_path", "audio": "base64_or_path", "context": {...}}
    """
    if not fusion_engine:
        raise HTTPException(status_code=503, detail="Multi-modal fusion not available")

    try:
        result = fusion_engine.process_multi_modal_input(
            inputs,
            context=inputs.get("context")
        )
        return result
    except Exception as e:
        logger.error(f"Multi-modal fusion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-modal fusion failed: {str(e)}")

@app.get("/api/multimodal/search")
async def search_multi_modal(query: str, modalities: Optional[List[str]] = None):
    """
    Search across multi-modal memories.

    Args:
        query: Search query
        modalities: Optional list of modalities to focus on
    """
    if not fusion_engine:
        raise HTTPException(status_code=503, detail="Multi-modal fusion not available")

    try:
        result = fusion_engine.search_multi_modal(query, modalities)
        return result
    except Exception as e:
        logger.error(f"Multi-modal search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-modal search failed: {str(e)}")

@app.post("/api/vision/compare")
async def compare_images(image1: Any, image2: Any):
    """
    Compare two images for similarity.

    Expected input: {"image1": "base64_or_path", "image2": "base64_or_path"}
    """
    if not vision_processor:
        raise HTTPException(status_code=503, detail="Vision processing not available")

    try:
        result = vision_processor.compare_images(image1, image2)
        return result
    except Exception as e:
        logger.error(f"Image comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image comparison failed: {str(e)}")

@app.post("/api/audio/compare")
async def compare_audio(audio1: Any, audio2: Any):
    """
    Compare two audio samples for similarity.

    Expected input: {"audio1": "base64_or_path", "audio2": "base64_or_path"}
    """
    if not audio_processor:
        raise HTTPException(status_code=503, detail="Audio processing not available")

    try:
        result = audio_processor.compare_audio(audio1, audio2)
        return result
    except Exception as e:
        logger.error(f"Audio comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio comparison failed: {str(e)}")

@app.get("/api/multimodal/stats")
async def get_multi_modal_stats():
    """Get statistics about multi-modal processing components."""
    stats = {
        "vision": {
            "available": vision_processor is not None,
            "cache_stats": vision_processor.get_cache_stats() if vision_processor else None
        },
        "audio": {
            "available": audio_processor is not None,
            "cache_stats": audio_processor.get_cache_stats() if audio_processor else None
        },
        "fusion": {
            "available": fusion_engine is not None,
            "fusion_stats": fusion_engine.get_fusion_stats() if fusion_engine else None
        }
    }
    return stats

# Model export endpoints (temporarily disabled - requires onnxruntime)
# @app.post("/api/models/export-mobile-bundle")
# async def export_mobile_bundle(bundle_name: str = "brein_mobile_v1"):
#     """Export models to mobile-friendly formats"""
#     exported_files = model_exporter.export_mobile_bundle(bundle_name)
#     return {"success": bool(exported_files), "files": exported_files}

# @app.post("/api/models/export-embedding")
# async def export_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
#     """Export embedding model to ONNX"""
#     path = model_exporter.export_embedding_model(model_name)
#     return {"success": path is not None, "path": path}

# @app.post("/api/models/export-transformer")
# async def export_memory_transformer():
#     """Export memory transformer to ONNX"""
#     path = model_exporter.export_memory_transformer()
#     return {"success": path is not None, "path": path}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Brein AI", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
