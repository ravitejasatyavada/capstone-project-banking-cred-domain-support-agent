# main.py
import sys
import numpy as np

# Critical Python 3.12 scalar alias boot compatibility fix hooks
if not hasattr(np, 'long'): np.long = int
if not hasattr(np, 'ulong'): np.ulong = int
sys.modules['numpy'].long = int
sys.modules['numpy'].ulong = int

import time
import uuid
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from pydantic import BaseModel, Field
from api.middleware import BudgetGatingMiddleware
from crew.crew_setup import execute_crew_workflow, run_input_guardrail
from review_stage import run_autogen_review

app = FastAPI(title="Cred Domain Support System")

# Mount the dedicated governance runtime budget cap middleware matrix layer cleanly
app.add_middleware(BudgetGatingMiddleware)

# Dedicated LFU/LRU Query Answer Mapping Cache Dictionary Object Reference
QUERY_CACHE = {}


class QueryRequest(BaseModel):
    query: str = Field(description="The primary policy question string submitted by the user.")
    record_id: str = Field(default=None, description="Optional unique customer loan database record reference key.")


@app.post("/ask")
async def ask_endpoint(payload: QueryRequest, response: Response) -> dict:
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    normalized_key = payload.query.strip().lower()

    # --- LRU CACHING DIAGNOSTICS LAYER ---
    if normalized_key in QUERY_CACHE:
        duration_ms = (time.time() - start_time) * 1000

        # Telemetry counter decrement log structure tracking performance optimization benchmarks
        print(json.dumps({
            "trace_id": trace_id,
            "metric": "cache_hit_diagnostic",
            "latency_ms": round(duration_ms, 4),
            "cache_state": "COUNTER_DECREMENT_SUCCESS"
        }))
        return {"source": "cache", "data": QUERY_CACHE[normalized_key]}

    try:
        masked_log_query = run_input_guardrail(payload.query)
        crew_out = execute_crew_workflow(payload.query, payload.record_id)
        final_verdict = run_autogen_review(crew_out.summary, f"Anchor Token: {payload.record_id}")

        output_payload = {
            "crew_status": crew_out.status,
            "quantum_inr": crew_out.loan_amount_inr,
            "escalation_index": crew_out.escalation_score,
            "verdict": final_verdict
        }

        QUERY_CACHE[normalized_key] = output_payload
        duration_ms = (time.time() - start_time) * 1000

        print(json.dumps({
            "trace_id": trace_id,
            "metric": "pipeline_miss_diagnostic",
            "latency_ms": round(duration_ms, 4),
            "query_masked": masked_log_query
        }))
        return {"source": "execution_engine", "data": output_payload}

    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@app.websocket("/chat")
async def chat_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_history = []
    try:
        while True:
            raw_text = await websocket.receive_text()
            data = json.loads(raw_text)
            session_history.append(f"User: {data.get('query', '')}")
            res = execute_crew_workflow(" | ".join(session_history), data.get('record_id'))
            session_history.append(f"Agent: {res.summary}")
            await websocket.send_json({"status": res.status, "summary": res.summary})
    except WebSocketDisconnect:
        pass
