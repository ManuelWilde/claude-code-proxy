"""Rate gate — fabric-level flow control for upstream LLM providers.

Solves three specific failure modes observed under 30–40 concurrent
Claude Code / aider / agent clients hitting a single upstream:

1. **Quota exhaustion storm.** Providers like z.ai return 429 with
   error code 1310 / "Weekly/Monthly Limit Exhausted". This is a hard
   error with a fixed reset timestamp; retrying it is wasted quota.
   The gate **circuit-breaks** for that upstream until the reset.

2. **Concurrency burst.** Each incoming request spawned its own
   outbound call → upstream saw a stampede → soft 429s rained down.
   The gate **caps concurrent in-flight requests per upstream** via an
   asyncio semaphore.

3. **Backoff ignoring `Retry-After`.** The gate honours upstream
   `Retry-After` hints before releasing the next request.

Configuration (all optional, safe defaults):

- ``FABRIC_RATE_GATE``            ``1`` to enable (default: enabled)
- ``FABRIC_RATE_MAX_INFLIGHT``    max concurrent per upstream (default 8)
- ``FABRIC_RATE_SOFT_BACKOFF_S``  floor for soft-429 retry delay
                                  (default 5.0)

Per MISSION.md §1.4: watchdog + declared budget + explicit failure
domain. The gate is the watchdog; the breaker is the fuse.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Regexes matching known hard-quota shapes we've seen in the wild.
# Extend conservatively: false-positives here mean users get told
# "quota exhausted" when they're actually soft-rate-limited.
_HARD_QUOTA_PATTERNS = [
    re.compile(r"weekly/monthly limit exhausted", re.IGNORECASE),
    re.compile(r"'code'\s*:\s*'1310'"),                 # z.ai
    re.compile(r"\b1310\b.*limit", re.IGNORECASE),
    re.compile(r"insufficient_quota", re.IGNORECASE),   # openai
    re.compile(r"monthly (quota|limit) (exceeded|exhausted)", re.IGNORECASE),
]

# "Your limit will reset at 2026-04-25 20:20:51" — naive timestamps.
_RESET_TS_PATTERN = re.compile(
    r"reset[^0-9]{0,20}"
    r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
)


@dataclass
class _BreakerState:
    open_until: float = 0.0          # epoch seconds
    last_reason: str = ""
    last_tripped_at: float = 0.0

    def is_open(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) < self.open_until


def _upstream_key(base_url: str) -> str:
    """Normalize a base_url to a stable per-host key."""
    try:
        p = urlparse(base_url)
        host = p.hostname or base_url
        return host.lower()
    except Exception:
        return base_url.lower()


def _parse_hard_quota(detail: str) -> Tuple[bool, Optional[float], str]:
    """If ``detail`` looks like a hard-quota exhaustion, return
    ``(True, reset_epoch_or_None, canonical_reason)``."""
    if not detail:
        return False, None, ""
    for pat in _HARD_QUOTA_PATTERNS:
        if pat.search(detail):
            reset_epoch: Optional[float] = None
            m = _RESET_TS_PATTERN.search(detail)
            if m:
                ts = m.group(1).replace("T", " ")
                try:
                    # Assume UTC if no tz info; providers are inconsistent here.
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    reset_epoch = dt.replace(tzinfo=timezone.utc).timestamp()
                except ValueError:
                    reset_epoch = None
            return True, reset_epoch, detail[:400]
    return False, None, ""


def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    """Parse an HTTP ``Retry-After`` header (seconds or HTTP-date)."""
    if not header_value:
        return None
    try:
        return float(header_value)
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(header_value)
        if dt is not None:
            return max(0.0, dt.timestamp() - time.time())
    except Exception:
        pass
    return None


class RateGate:
    """Per-upstream semaphore + circuit breaker singleton."""

    _instance: "Optional[RateGate]" = None

    def __init__(self,
                 max_inflight: int = 8,
                 soft_backoff_s: float = 5.0,
                 enabled: bool = True):
        self.enabled = enabled
        self.max_inflight = max(1, max_inflight)
        self.soft_backoff_s = max(0.0, soft_backoff_s)
        self._sems: Dict[str, asyncio.Semaphore] = {}
        self._breakers: Dict[str, _BreakerState] = {}
        self._sems_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Singleton access (configured from env on first use)
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "RateGate":
        if cls._instance is None:
            cls._instance = cls(
                max_inflight=int(os.environ.get("FABRIC_RATE_MAX_INFLIGHT", "8")),
                soft_backoff_s=float(os.environ.get("FABRIC_RATE_SOFT_BACKOFF_S", "5.0")),
                enabled=os.environ.get("FABRIC_RATE_GATE", "1") != "0",
            )
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def acquire(self, base_url: str) -> "asyncio.Semaphore":
        """Acquire an in-flight slot for ``base_url``. Raises
        :class:`QuotaExhausted` if the breaker is open."""
        key = _upstream_key(base_url)
        if self.enabled:
            br = self._breakers.get(key)
            if br and br.is_open():
                raise QuotaExhausted(
                    upstream=key,
                    reset_at=br.open_until,
                    reason=br.last_reason,
                )
        sem = await self._get_sem(key)
        await sem.acquire()
        return sem

    def release(self, sem: "asyncio.Semaphore") -> None:
        try:
            sem.release()
        except ValueError:
            pass

    def trip_quota(self, base_url: str, reset_epoch: Optional[float], reason: str) -> None:
        """Open the breaker until ``reset_epoch`` (or +1 h fallback)."""
        key = _upstream_key(base_url)
        open_until = reset_epoch if (reset_epoch and reset_epoch > time.time()) else (time.time() + 3600)
        self._breakers[key] = _BreakerState(
            open_until=open_until,
            last_reason=reason,
            last_tripped_at=time.time(),
        )
        logger.error(
            "rate-gate: QUOTA breaker OPEN for %s until %s (%.0fs). reason=%s",
            key,
            datetime.fromtimestamp(open_until, tz=timezone.utc).isoformat(),
            open_until - time.time(),
            reason[:200],
        )

    def suggest_soft_backoff(self, retry_after_header: Optional[str] = None) -> float:
        """Return the floor delay to use for a soft-429 retry."""
        ra = _parse_retry_after(retry_after_header)
        if ra is not None:
            return max(ra, 0.5)
        return self.soft_backoff_s

    def status(self) -> Dict[str, Dict[str, str]]:
        """Debug view of breaker/semaphore state."""
        now = time.time()
        out: Dict[str, Dict[str, str]] = {}
        for key, br in self._breakers.items():
            out[key] = {
                "breaker_open": str(br.is_open(now)),
                "open_until": datetime.fromtimestamp(br.open_until, tz=timezone.utc).isoformat(),
                "reason": br.last_reason[:200],
            }
        for key, sem in self._sems.items():
            out.setdefault(key, {})["inflight_slots_total"] = str(self.max_inflight)
        return out

    # ------------------------------------------------------------------
    async def _get_sem(self, key: str) -> "asyncio.Semaphore":
        sem = self._sems.get(key)
        if sem is not None:
            return sem
        async with self._sems_lock:
            sem = self._sems.get(key)
            if sem is None:
                sem = asyncio.Semaphore(self.max_inflight)
                self._sems[key] = sem
        return sem


class QuotaExhausted(Exception):
    """Raised when the breaker is open for an upstream."""

    def __init__(self, upstream: str, reset_at: float, reason: str):
        self.upstream = upstream
        self.reset_at = reset_at
        self.reason = reason
        delta = max(0, int(reset_at - time.time()))
        reset_iso = datetime.fromtimestamp(reset_at, tz=timezone.utc).isoformat()
        super().__init__(
            f"quota exhausted for {upstream}; resets at {reset_iso} "
            f"(~{delta}s). reason={reason[:200]}"
        )


def classify_429(error_detail: str) -> Tuple[bool, Optional[float], str]:
    """Public wrapper over :func:`_parse_hard_quota` for use in callers."""
    return _parse_hard_quota(error_detail)
