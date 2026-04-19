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
- ``FABRIC_RATE_PROBE_INTERVAL_S`` while the quota breaker is open, allow
                                  one probe every N seconds (half-open
                                  state). If the probe succeeds, close
                                  the breaker immediately. If it fails
                                  with another hard quota, stay open for
                                  another interval. (default 60.0)

Per MISSION.md §1.4: watchdog + declared budget + explicit failure
domain. The gate is the watchdog; the breaker is the fuse.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse


# Trust-but-verify window: require multiple hard-429 signals within this
# many seconds before actually tripping the breaker. Upstream error
# messages are noisy / sometimes wrong; one "Quota Exhausted" claim is
# not enough evidence when the next request might succeed.
CONFIRM_WINDOW_S = 30.0
MIN_CONFIRMATIONS = 2
# Even when we DO trust a declared reset timestamp, we cap the probe
# interval so we re-check well before that. If reset is in 5 days we
# still probe every minute so a bursty quota recovery is picked up.
MAX_OPEN_WITHOUT_PROBE_S = 60.0

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
    open_until: float = 0.0          # epoch seconds; 0 = closed
    last_reason: str = ""
    last_tripped_at: float = 0.0
    last_probe_at: float = 0.0
    probe_inflight: bool = False     # guards against concurrent probes
    # Trust-but-verify: hard-429 observations form a sliding window; we
    # only trip the breaker once the signal is confirmed (default: ≥2 in
    # CONFIRM_WINDOW_S). Stale entries are pruned on each report.
    recent_hard_429s: list = field(default_factory=list)  # type: ignore[var-annotated]

    def is_open(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) < self.open_until

    def can_probe(self, interval_s: float, now: Optional[float] = None) -> bool:
        """True if the breaker is open and it's time to send one probe."""
        t = now or time.time()
        return (
            self.is_open(t)
            and not self.probe_inflight
            and (t - self.last_probe_at) >= interval_s
        )


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
                 probe_interval_s: float = 60.0,
                 enabled: bool = True):
        self.enabled = enabled
        self.max_inflight = max(1, max_inflight)
        self.soft_backoff_s = max(0.0, soft_backoff_s)
        self.probe_interval_s = max(1.0, probe_interval_s)
        self._sems: Dict[str, asyncio.Semaphore] = {}
        self._breakers: Dict[str, _BreakerState] = {}
        self._sems_lock = asyncio.Lock()
        self._breakers_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Singleton access (configured from env on first use)
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "RateGate":
        if cls._instance is None:
            cls._instance = cls(
                max_inflight=int(os.environ.get("FABRIC_RATE_MAX_INFLIGHT", "8")),
                soft_backoff_s=float(os.environ.get("FABRIC_RATE_SOFT_BACKOFF_S", "5.0")),
                probe_interval_s=float(os.environ.get("FABRIC_RATE_PROBE_INTERVAL_S", "60.0")),
                enabled=os.environ.get("FABRIC_RATE_GATE", "1") != "0",
            )
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def acquire(self, base_url: str) -> Tuple["asyncio.Semaphore", bool]:
        """Acquire an in-flight slot for ``base_url``.

        Returns ``(sem, is_probe)``. If the breaker is open and it's time
        for the next half-open probe, returns ``(sem, True)`` so exactly
        ONE request gets through; the caller is responsible for reporting
        the outcome via :meth:`note_success` or :meth:`note_hard_429`.
        Raises :class:`QuotaExhausted` if the breaker is open and no
        probe slot is available yet.
        """
        key = _upstream_key(base_url)
        is_probe = False
        if self.enabled:
            br = self._breakers.get(key)
            if br and br.is_open():
                if br.can_probe(self.probe_interval_s):
                    # Claim the probe slot atomically.
                    async with self._breakers_lock:
                        br = self._breakers.get(key)
                        if br and br.is_open() and br.can_probe(self.probe_interval_s):
                            br.probe_inflight = True
                            br.last_probe_at = time.time()
                            is_probe = True
                            logger.info(
                                "rate-gate: sending probe to %s (breaker open, half-open)",
                                key,
                            )
                        else:
                            raise QuotaExhausted(
                                upstream=key,
                                reset_at=br.open_until if br else time.time(),
                                reason=(br.last_reason if br else ""),
                            )
                else:
                    raise QuotaExhausted(
                        upstream=key,
                        reset_at=br.open_until,
                        reason=br.last_reason,
                    )
        sem = await self._get_sem(key)
        await sem.acquire()
        return sem, is_probe

    def release(self, sem: "asyncio.Semaphore") -> None:
        try:
            sem.release()
        except ValueError:
            pass

    # -----------------------------------------------------------------
    # Outcome reporting (trust-but-verify)
    # -----------------------------------------------------------------
    def note_success(self, base_url: str) -> None:
        """A 2xx response just came back. Strong evidence the upstream
        is healthy: clear the breaker if open, drop stale hard-429
        observations. Always safe to call."""
        key = _upstream_key(base_url)
        br = self._breakers.get(key)
        if br is None:
            return
        was_open = br.is_open()
        if was_open:
            logger.warning(
                "rate-gate: probe succeeded for %s; CLOSING breaker "
                "(declared reset was %s). Upstream message was apparently stale.",
                key,
                datetime.fromtimestamp(br.open_until, tz=timezone.utc).isoformat(),
            )
        # Drop the breaker entirely.
        self._breakers.pop(key, None)

    def note_hard_429(self,
                      base_url: str,
                      reset_epoch: Optional[float],
                      reason: str,
                      was_probe: bool = False) -> bool:
        """Record a hard-quota observation. Returns True if this
        observation tripped (or kept) the breaker open, False if the
        system is still in 'single noisy message, not yet confirmed'
        state.

        Trust-but-verify policy:
        - A single hard-429 is *noise* until confirmed by a second one
          inside ``CONFIRM_WINDOW_S``. Until confirmed, the breaker
          stays closed and normal retry/backoff applies.
        - Two hard-429s inside the window are confirmation: open the
          breaker, but cap the probe-free window at
          ``MAX_OPEN_WITHOUT_PROBE_S`` so we re-check aggressively
          regardless of how far out the upstream claims reset is.
        - If this observation is a probe outcome, it always counts as
          confirmation — we already had prior evidence when we opened
          the breaker, and the probe re-confirmed.
        """
        key = _upstream_key(base_url)
        now = time.time()
        br = self._breakers.setdefault(key, _BreakerState())

        # Probe outcome: clear probe_inflight, keep breaker open, extend.
        if was_probe:
            br.probe_inflight = False
            br.last_probe_at = now
            br.last_reason = reason[:400]
            br.recent_hard_429s = [now]  # reset window; we know current state
            # Keep open_until up-to-date but capped.
            declared = reset_epoch if (reset_epoch and reset_epoch > now) else (now + 3600)
            br.open_until = min(declared, now + max(self.probe_interval_s, MAX_OPEN_WITHOUT_PROBE_S))
            logger.warning(
                "rate-gate: probe failed with hard-429 for %s; staying OPEN for %.0fs",
                key, br.open_until - now,
            )
            return True

        # Normal (non-probe) observation — run the confirmation filter.
        br.recent_hard_429s = [t for t in br.recent_hard_429s if now - t <= CONFIRM_WINDOW_S]
        br.recent_hard_429s.append(now)

        if br.is_open():
            # Already tripped; keep it so, refresh bound.
            br.last_reason = reason[:400]
            declared = reset_epoch if (reset_epoch and reset_epoch > now) else (now + 3600)
            br.open_until = min(declared, now + max(self.probe_interval_s, MAX_OPEN_WITHOUT_PROBE_S))
            return True

        if len(br.recent_hard_429s) < MIN_CONFIRMATIONS:
            logger.info(
                "rate-gate: hard-429 observed for %s (1/%d) — NOT tripping yet; upstream messages are not always true.",
                key, MIN_CONFIRMATIONS,
            )
            return False

        # Confirmed → trip.
        declared = reset_epoch if (reset_epoch and reset_epoch > now) else (now + 3600)
        br.open_until = min(declared, now + max(self.probe_interval_s, MAX_OPEN_WITHOUT_PROBE_S))
        br.last_reason = reason[:400]
        br.last_tripped_at = now
        logger.error(
            "rate-gate: QUOTA breaker OPEN for %s (confirmed %d times within %.0fs). "
            "Will probe every %.0fs. Declared reset: %s.",
            key, len(br.recent_hard_429s), CONFIRM_WINDOW_S, self.probe_interval_s,
            datetime.fromtimestamp(declared, tz=timezone.utc).isoformat(),
        )
        return True

    def note_probe_connection_error(self, base_url: str) -> None:
        """Probe raised a connection error (network, not upstream
        verdict). Free the probe slot but keep breaker state — a broken
        network doesn't tell us anything about quota."""
        key = _upstream_key(base_url)
        br = self._breakers.get(key)
        if br is not None:
            br.probe_inflight = False
            br.last_probe_at = time.time()

    # Back-compat thin wrapper (old callers).
    def trip_quota(self, base_url: str, reset_epoch: Optional[float], reason: str) -> None:
        self.note_hard_429(base_url, reset_epoch, reason, was_probe=False)

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
