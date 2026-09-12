# api/middleware.py
import json
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class BudgetGatingMiddleware(BaseHTTPMiddleware):
    """
    Enforces runtime transaction budget caps by intercepting incoming payload envelopes.
    Rejects oversized payloads before passing traffic to downstream multi-agent execution layers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Intercept POST transactions targeting the core agent interface route
        if request.url.path == "/ask" and request.method == "POST":
            # Read structural envelope dimensions out of the standard HTTP content headers
            content_length = request.headers.get("content-length")

            if content_length is not None:
                # Gating Gate: Throw an explicit 402 simulation budget cap error if size crosses threshold bounds
                if int(content_length) > 1000000:  # 1 Megabyte Scale Cap Buffer Limit
                    return Response(
                        content=json.dumps({
                                               "detail": "Budget cap breached: Transaction footprint exceeds simulated token allowance bounds."}),
                        status_code=402,  # Explicit HTTP 402 Simulation Error mandated by review checklist
                        media_type="application/json"
                    )

        return await call_next(request)
