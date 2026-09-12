# api/middleware.py
import time
import json
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Centralised, high-performance runtime sliding-window storage tracking ip addresses
RATE_LIMIT_RECORD = {}
CALL_LIMIT_WINDOW_SECS = 60
MAX_PERMITTED_CALLS = 30


class GovernanceShieldMiddleware(BaseHTTPMiddleware):
    """
    Unified AI Governance and System Resilience Middleware Layer.

    Enforces two strict resilience layers:
    1. Runtime Layer Budget Cap: Catches and drops oversized content sizes (>1MB) with a 402 error.
    2. Operational Rate Limiting: Limits connection traffic to 30 requests per minute per IP address,
       returning a 429 error if breached, protecting caching layers and agent loops alike.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Intercept POST transactions targeting the core agent interface route
        if request.url.path == "/ask" and request.method == "POST":
            client_ip = request.client.host
            current_time = time.time()

            # -----------------------------------------------------------------
            # GUARDRAIL LAYER 1: OPERATIONAL RATE LIMITING CAP
            # -----------------------------------------------------------------
            # Housekeep and clear out expired window timestamps older than 60 seconds
            timestamps = RATE_LIMIT_RECORD.get(client_ip, [])
            timestamps = [t for t in timestamps if current_time - t < CALL_LIMIT_WINDOW_SECS]
            RATE_LIMIT_RECORD[client_ip] = timestamps

            # Gating check: Instantly reject request if the client exceeds the 30-call boundary ceiling
            if len(timestamps) >= MAX_PERMITTED_CALLS:
                return Response(
                    content=json.dumps({"detail": "Too many requests. Security threshold enforced."}),
                    status_code=429,  # Mandated HTTP 429 security trip code
                    media_type="application/json"
                )

            # Log the active current timestamp to the sliding window track matrix
            RATE_LIMIT_RECORD[client_ip].append(current_time)

            # -----------------------------------------------------------------
            # GUARDRAIL LAYER 2: RUNTIME CONTENT FOOTPRINT BUDGET CAP
            # -----------------------------------------------------------------
            content_length = request.headers.get("content-length")
            if content_length is not None:
                if int(content_length) > 1000000:  # 1 Megabyte Scale Cap Buffer Limit
                    return Response(
                        content=json.dumps({
                                               "detail": "Budget cap breached: Transaction footprint exceeds simulated token allowance bounds."}),
                        status_code=402,  # Explicit HTTP 402 Simulation Error mandated by review checklist
                        media_type="application/json"
                    )

        return await call_next(request)
