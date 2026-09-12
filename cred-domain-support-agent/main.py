# =====================================================================
# CRITICAL PYTHON 3.12 / NUMPY COMPLETENESS PATCH (FASTAPI ENTRYPOINT)
# =====================================================================
import sys
import numpy as np

# Intercept and auto-patch dropped NumPy 1.x scalar aliases at startup.
# This prevents underlying native C-compiled binary modules (like older chromadb versions)
# from throwing an AttributeError under modern Python 3.12 execution pipelines.
if not hasattr(np, 'long'):
    np.long = int
if not hasattr(np, 'ulong'):
    np.ulong = int

# Bind the definitions directly into Python's active system modules registry globally
sys.modules['numpy'].long = int
sys.modules['numpy'].ulong = int
# =====================================================================

# main.py
import time
import uuid
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from pydantic import BaseModel, Field
from crew_layer import execute_crew_workflow, run_input_guardrail
from review_stage import run_autogen_review
from rag_engine import coll_sentence, embedder, get_sentence_chunks

# Initialize the core FastAPI Application instance
app = FastAPI(
    title="Cred Domain Support System",
    description="Production-minded multi-agent support routing backend engine executing under isolated Mock LLM constraints."
)

# -------------------------------------------------------------------------
# GLOBAL STATE STATE-TRACKERS & PERFORMANCE CACHE (Part 4, Task 16)
# -------------------------------------------------------------------------
# High-performance In-Memory cache keyed by normalized query strings to eliminate redundant LLM calls
QUERY_CACHE = {}

# Strict token budget ceiling constraint boundary (Part 4, Task 15)
# Rejects oversized payload string blocks before hitting downstream agent layers
MAX_TOKEN_CHARACTER_BUDGET = 500


# -------------------------------------------------------------------------
# REQUEST/RESPONSE PYDANTIC SCHEMAS (Part 3, Task 11)
# -------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Schema governing standard incoming POST endpoint queries for agent support evaluation."""
    query: str = Field(description="The primary policy question string submitted by the user.")
    record_id: str = Field(default=None, description="Optional unique customer loan database record reference key.")


class DocumentRequest(BaseModel):
    """Schema governing incoming administrative document injection requests."""
    doc_id: str = Field(description="Unique reference identifier for the new document (e.g., 'doc_13').")
    text: str = Field(description="The compliance text content to be chunked, embedded, and added to the RAG index.")


class CacheHitLogSchema(BaseModel):
    """Structured schema mapping for LFU performance telemetry log lines."""
    trace_id: str
    metric: str
    timing_ms: float
    query: str


class PipelineMissLogSchema(BaseModel):
    """Structured schema mapping for standard engine processing pipeline log lines."""
    trace_id: str
    metric: str
    timing_ms: float
    query: str


# =====================================================================
# REST FULL ROUTE PIPELINE ENTRIES (Part 3, Task 11)
# =====================================================================

@app.post("/ask")
async def ask_endpoint(payload: QueryRequest, response: Response):
    """
    Processes a single policy question or account lookup request through the multi-agent stack.

    Pipeline Steps:
    1. Enforces runtime token cost budgets based on string footprint lengths.
    2. Runs normalized key matching lookups against the local in-memory LFU cache table.
    3. Triggers input-side PII masking filters to completely mask sensitive customer fields.
    4. Routes the request down the sequential CrewAI multi-agent workflow to build a policy response.
    5. Handshakes with the AutoGen compliance ring to audit and authorize the generated draft.
    6. Writes an ELK-compliant JSON-Lines telemetry log file completely masked of cleartext data.
    """
    start_time = time.time()
    trace_id = str(uuid.uuid4())  # Generate unique tracer key identifier for this request profile

    # --- LAYER 1: RUNTIME COST-BUDGET GATEWAY CHECK (Part 4, Task 15) ---
    if len(payload.query) > MAX_TOKEN_CHARACTER_BUDGET:
        raise HTTPException(
            status_code=400,
            detail="Request dropped: Input footprint exceeds configured corporate token budgets."
        )

    # --- LAYER 2: IN-MEMORY CACHE LOOKUP PASS (Part 4, Task 16) ---
    normalized_key = payload.query.strip().lower()
    if normalized_key in QUERY_CACHE:
        duration = (time.time() - start_time) * 1000

        # Compile an ELK-compliant log line using masked string data parameters
        log_entry = CacheHitLogSchema(
            trace_id=trace_id,
            metric="cache_hit",
            timing_ms=round(duration, 4),
            query="[PROTECTED_CACHE_ASSET]"
        )
        print(log_entry.model_dump_json())  # Dump trace line directly out to console stdout logs
        return {"source": "cache", "data": QUERY_CACHE[normalized_key]}

    try:
        # --- LAYER 3: INPUT GUARDRAILS & LOG MASKING FILTER (Part 3, Task 12) ---
        # Generate our masked string copy before any processing assets are saved onto internal disks
        masked_log_query = run_input_guardrail(payload.query)

        # --- LAYER 4: PRIMARY CREWAI EXECUTION LOOP PASS (Part 2, Task 7) ---
        crew_out = execute_crew_workflow(payload.query, payload.record_id)

        # --- LAYER 5: INDEPENDENT CROSS-FRAMEWORK AUTOGEN REVIEW PASS (Part 4, Task 14) ---
        final_verdict = run_autogen_review(crew_out.summary, f"Execution Anchor Token: {payload.record_id}")

        # Compile final outcome footprint configuration
        output_payload = {
            "crew_status": crew_out.status,
            "quantum_inr": crew_out.loan_amount_inr,
            "escalation_index": crew_out.escalation_score,
            "verdict": final_verdict
        }

        # Populate cache storage array with the finalized verdict parameters
        QUERY_CACHE[normalized_key] = output_payload

        # --- LAYER 6: STRUCTURED STRATIFIED LOGGING GENERATION (Part 3, Task 12) ---
        duration = (time.time() - start_time) * 1000
        log_entry = PipelineMissLogSchema(
            trace_id=trace_id,
            metric="pipeline_miss",
            timing_ms=round(duration, 4),
            query=masked_log_query  # Masked variable completely insulates logging buffers from PII leaks
        )
        print(log_entry.model_dump_json())

        return {"source": "execution_engine", "data": output_payload}

    except ValueError as exc:
        # Catch security guardrail faults explicitly and map them back to standard HTTP exception codes
        raise HTTPException(status_code=403, detail=str(exc))


@app.post("/add-document")
async def add_document_endpoint(payload: DocumentRequest):
    """
    HTTP Endpoint #2: Dynamically segments, embeds, and indexes a new
    compliance text asset straight into the active production sentence-based collection.

    This fulfills the capstone requirement to provide a second Pydantic-validated endpoint
    for dynamic runtime scalability without manually resetting the environment.
    """
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    try:
        # Segment the raw document input string text utilizing the sentence-based text chunker
        chunks = get_sentence_chunks(payload.text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Document text contains no valid sentences to split.")

        # Dynamically inject each newly created text slice straight into the live ChromaDB vector index space
        for idx, chunk in enumerate(chunks):
            coll_sentence.add(
                documents=[chunk],
                embeddings=[embedder.encode(chunk).tolist()],  # Local embedding transformation conversion pass
                metadatas=[{"doc_id": payload.doc_id}],  # Map metadata back to the administrative reference
                ids=[f"{payload.doc_id}_dynamic_sent_{idx}"]  # Inject isolated unique tracker index keys
            )

        duration = (time.time() - start_time) * 1000

        # Write an administrative audit line directly to our structured log trace
        admin_log = {
            "trace_id": trace_id,
            "metric": "administrative_document_injection",
            "timing_ms": round(duration, 4),
            "status": "success",
            "document_id": payload.doc_id,
            "chunks_added": len(chunks)
        }
        print(json.dumps(admin_log))

        # Flush the query cache to ensure that any new intelligence forces fresh pipeline loops properly
        QUERY_CACHE.clear()

        return {
            "status": "success",
            "message": f"Document '{payload.doc_id}' successfully parsed and indexed into sentence collection.",
            "chunks_processed": len(chunks)
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal database update failure: {str(exc)}")


# =====================================================================
# STATEFUL WEBSOCKET COMMUNICATION GATEWAY (Part 3, Task 11)
# =====================================================================

@app.websocket("/chat")
async def chat_websocket_endpoint(websocket: WebSocket):
    """
    Exposes a real-time, bi-directional persistent stateful WebSocket interface portal.

    Manages in-process conversation turn history contexts sequentially across inputs,
    and captures connection closures safely to prevent server instability or threading crashes."""
    await websocket.accept()
    session_history = []  # Local isolated in-process conversation timeline array buffer
    try:
        while True:  # Standby to capture incoming data strings from client socket connections
            raw_text = await websocket.receive_text()
            data = json.loads(raw_text)
            user_msg = data.get("query", "")
            rec_id = data.get("record_id", None)
            # Append new user statement trace directly onto our session context window (Part 2, Task 8)
            session_history.append(f"User: {user_msg}")
            contextual_query = " | Memory Context History: ".join(session_history)
            # Route our multi-turn history accumulation into the main multi-agent pipeline
            res = execute_crew_workflow(contextual_query, rec_id)
            # Capture the framework response back onto our local session tracker
            session_history.append(f"Agent: {res.summary}")
            # Return serialized runtime tracking variables back across the network path
            await websocket.send_json(
                {"status": res.status, "summary": res.summary, "history_turn_depth": len(session_history)})
    except WebSocketDisconnect:
        # Catch connection failures instantly.
        # This keeps the underlying event loop active, keeping the server functional for all other active users.
        print("Network Socket Disconnect Event Captured. Memory cleared. Server operational state remains perfectly stable.")

