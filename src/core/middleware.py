"""Middleware: request size limits and rate limiting."""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding a configured size."""

    def __init__(self, app: FastAPI, max_body_bytes: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_bytes:
            return Response(
                content='{"detail":"Request body too large"}',
                status_code=413,
                media_type="application/json",
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory per-IP rate limiting using a sliding window."""

    _PRUNE_INTERVAL = 100  # prune stale IPs every N requests

    def __init__(
        self,
        app: FastAPI,
        max_requests: int = 60,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._request_count = 0

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _prune_stale_ips(self, cutoff: float):
        """Remove IPs with no recent activity."""
        stale = [ip for ip, ts in self._requests.items() if not ts or ts[-1] < cutoff]
        for ip in stale:
            del self._requests[ip]

    def _is_limited(self, ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        timestamps = self._requests[ip]
        self._requests[ip] = timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= self.max_requests:
            return True
        timestamps.append(now)

        # Periodically prune stale IPs to prevent unbounded memory growth
        self._request_count += 1
        if self._request_count >= self._PRUNE_INTERVAL:
            self._request_count = 0
            self._prune_stale_ips(cutoff)

        return False

    async def dispatch(self, request: Request, call_next):
        ip = self._client_ip(request)
        if self._is_limited(ip):
            logger.warning(f"Rate limit exceeded for {ip}")
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )
        return await call_next(request)
