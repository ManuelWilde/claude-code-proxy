import asyncio
import json
import logging
import random
import time
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError, APIConnectionError, APITimeoutError

from src.core.rate_gate import RateGate, QuotaExhausted, classify_429

logger = logging.getLogger(__name__)


STREAM_CHUNK_TIMEOUT = 120  # seconds between chunks before considering stream stalled


class OpenAIClient:
    """Async OpenAI client with cancellation and retry support."""

    def __init__(self, api_key: str, base_url: str, timeout: int = 90, api_version: Optional[str] = None, custom_headers: Optional[Dict[str, str]] = None, max_retries: int = 2):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        self.max_retries = max_retries

        # Prepare default headers
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "claude-proxy/1.0.0"
        }

        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}

        # Detect if using Azure and instantiate the appropriate client.
        # max_retries=0: the openai SDK's internal retry is disabled so the
        # fabric RateGate is the sole retry authority. Without this, the SDK
        # silently retries 429s at the HTTP layer before our quota breaker
        # ever sees them, burning tokens against an already-exhausted quota.
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                max_retries=0,
                default_headers=all_headers
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
                default_headers=all_headers
            )
        self.active_requests: Dict[str, asyncio.Event] = {}

    def _is_retriable(self, error: Exception) -> bool:
        """Check if an error is worth retrying."""
        return isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError))

    async def _execute_with_retry(self, coro_factory, operation: str = "request"):
        """Execute an async callable behind the fabric RateGate.

        - Acquires a per-upstream semaphore slot (caps fleet-wide concurrency).
        - Fails fast with HTTP 429 if the upstream's quota breaker is open.
        - On soft 429, honours ``Retry-After`` and a configurable floor instead
          of hammering with a hardcoded 1-2s backoff.
        - On hard quota exhaustion (z.ai code 1310, OpenAI ``insufficient_quota``,
          etc.) trips the breaker so subsequent requests short-circuit until
          the declared reset timestamp.
        """
        gate = RateGate.instance()
        last_error = None

        # Breaker short-circuit before we even reach for the semaphore.
        # The gate may grant a half-open probe slot if the breaker is
        # open but due for re-check; is_probe tells us to report the
        # outcome with probe=True so the breaker resolves correctly.
        try:
            sem, is_probe = await gate.acquire(self.base_url)
        except QuotaExhausted as qx:
            raise HTTPException(status_code=429, detail=str(qx))

        try:
            for attempt in range(self.max_retries + 1):
                try:
                    result = await coro_factory()
                except AuthenticationError as e:
                    raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
                except BadRequestError as e:
                    raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
                except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                    last_error = e
                    detail_str = str(e)

                    # Hard quota? Don't trust a single message — route it
                    # through the gate's trust-but-verify policy.
                    if isinstance(e, RateLimitError):
                        hard, reset_epoch, reason = classify_429(detail_str)
                        if hard:
                            tripped = gate.note_hard_429(
                                self.base_url,
                                reset_epoch,
                                reason,
                                was_probe=is_probe,
                            )
                            if tripped:
                                raise HTTPException(
                                    status_code=429,
                                    detail=f"Quota exhausted upstream: {self.classify_openai_error(detail_str)}",
                                )
                            # Unconfirmed: treat as soft 429 and retry.

                    # Probe connection errors don't prove anything about
                    # upstream quota; let the gate release the probe slot
                    # so the next probe can be tried sooner.
                    if is_probe and isinstance(e, (APIConnectionError, APITimeoutError)):
                        gate.note_probe_connection_error(self.base_url)

                    if attempt < self.max_retries:
                        retry_after = getattr(getattr(e, "response", None), "headers", {}) or {}
                        ra_hdr = retry_after.get("Retry-After") if hasattr(retry_after, "get") else None
                        soft_floor = gate.suggest_soft_backoff(ra_hdr)
                        base_delay = max(soft_floor, (2 ** attempt) + (attempt * 0.5))
                        jitter = random.uniform(0, 0.5)
                        delay = base_delay + jitter
                        logger.warning(
                            "%s soft-429/conn (attempt %d/%d) upstream=%s retry in %.1fs: %s",
                            operation, attempt + 1, self.max_retries + 1,
                            self.base_url, delay, detail_str[:200],
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        if isinstance(e, RateLimitError):
                            raise HTTPException(status_code=429, detail=self.classify_openai_error(detail_str))
                        elif isinstance(e, APIConnectionError):
                            raise HTTPException(status_code=502, detail=f"Connection error: {self.classify_openai_error(detail_str)}")
                        elif isinstance(e, APITimeoutError):
                            raise HTTPException(status_code=504, detail=f"Request timed out: {self.classify_openai_error(detail_str)}")
                        raise HTTPException(status_code=502, detail=self.classify_openai_error(detail_str))
                except APIError as e:
                    status_code = getattr(e, 'status_code', None) or 500
                    raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

                # Success path: call closes the breaker if it was open
                # (strong evidence upstream is healthy; declared reset
                # was apparently stale).
                gate.note_success(self.base_url)
                return result
        finally:
            gate.release(sem)
    
    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation and retry support."""

        cancel_event = None
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            async def _do_request():
                completion_task = asyncio.create_task(
                    self.client.chat.completions.create(**request)
                )

                if cancel_event is not None:
                    cancel_task = asyncio.create_task(cancel_event.wait())
                    try:
                        done, pending = await asyncio.wait(
                            [completion_task, cancel_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        for task in pending:
                            task.cancel()

                        if cancel_task in done:
                            completion_task.cancel()
                            raise HTTPException(status_code=499, detail="Request cancelled by client")

                        return completion_task.result()
                    except HTTPException:
                        raise
                    except Exception:
                        # Ensure task is cancelled on any unexpected error
                        completion_task.cancel()
                        raise
                    finally:
                        # Clean up cancel_task
                        if not cancel_task.done():
                            cancel_task.cancel()
                else:
                    return await completion_task

            completion = await self._execute_with_retry(_do_request, "chat_completion")
            return completion.model_dump()

        finally:
            if request_id:
                self.active_requests.pop(request_id, None)
    
    async def create_chat_completion_stream(self, request: Dict[str, Any], request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation and retry support."""

        cancel_event = None
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            stream_request = {**request, "stream": True, "stream_options": {"include_usage": True}}

            streaming_completion = await self._execute_with_retry(
                lambda: self.client.chat.completions.create(**stream_request),
                "stream_completion"
            )

            async def _chunk_iter():
                async for chunk in streaming_completion:
                    yield chunk

            async for chunk in _chunk_iter():
                if cancel_event is not None and cancel_event.is_set():
                    raise HTTPException(status_code=499, detail="Request cancelled by client")

                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"

            yield "data: [DONE]"

        finally:
            if request_id:
                self.active_requests.pop(request_id, None)

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."
        
        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False