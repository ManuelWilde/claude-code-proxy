import asyncio
import json
import logging
import time
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError, APIConnectionError, APITimeoutError

logger = logging.getLogger(__name__)

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
        
        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=all_headers
            )
        self.active_requests: Dict[str, asyncio.Event] = {}

    def _is_retriable(self, error: Exception) -> bool:
        """Check if an error is worth retrying."""
        return isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError))

    async def _execute_with_retry(self, coro_factory, operation: str = "request"):
        """Execute an async callable with exponential backoff retry."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_factory()
            except AuthenticationError as e:
                raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
            except BadRequestError as e:
                raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = (2 ** attempt) + (attempt * 0.5)
                    logger.warning(f"{operation} failed (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    if isinstance(e, RateLimitError):
                        raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
                    elif isinstance(e, APIConnectionError):
                        raise HTTPException(status_code=502, detail=f"Connection error: {self.classify_openai_error(str(e))}")
                    elif isinstance(e, APITimeoutError):
                        raise HTTPException(status_code=504, detail=f"Request timed out: {self.classify_openai_error(str(e))}")
                    raise HTTPException(status_code=502, detail=self.classify_openai_error(str(e)))
            except APIError as e:
                status_code = getattr(e, 'status_code', None) or 500
                raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation and retry support."""

        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            async def _do_request():
                completion_task = asyncio.create_task(
                    self.client.chat.completions.create(**request)
                )

                if request_id:
                    cancel_task = asyncio.create_task(cancel_event.wait())
                    done, pending = await asyncio.wait(
                        [completion_task, cancel_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    if cancel_task in done:
                        completion_task.cancel()
                        raise HTTPException(status_code=499, detail="Request cancelled by client")

                    return await completion_task
                else:
                    return await completion_task

            completion = await self._execute_with_retry(_do_request, "chat_completion")
            return completion.model_dump()

        finally:
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def create_chat_completion_stream(self, request: Dict[str, Any], request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation and retry support."""

        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            request["stream"] = True
            if "stream_options" not in request:
                request["stream_options"] = {}
            request["stream_options"]["include_usage"] = True

            streaming_completion = await self._execute_with_retry(
                lambda: self.client.chat.completions.create(**request),
                "stream_completion"
            )

            async for chunk in streaming_completion:
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(status_code=499, detail="Request cancelled by client")

                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"

            yield "data: [DONE]"

        finally:
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

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