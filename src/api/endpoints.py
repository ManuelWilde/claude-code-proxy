import traceback
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.config import Config
from src.core.client import OpenAIClient
from src.core.client_registry import ClientRegistry
from src.core.model_manager import ModelManager
from src.core.dependencies import get_config, get_client_registry, get_model_manager
from src.core.logging import logger
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude,
)

router = APIRouter()


async def validate_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    config = get_config(request)
    client_api_key = None

    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization[7:]

    if not config.anthropic_api_key:
        return

    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning("Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key.",
        )


@router.post("/v1/messages")
async def create_message(
    request: ClaudeMessagesRequest,
    http_request: Request,
    _auth: None = Depends(validate_api_key),
    config: Config = Depends(get_config),
    registry: ClientRegistry = Depends(get_client_registry),
    model_mgr: ModelManager = Depends(get_model_manager),
):
    try:
        logger.debug(
            f"Processing Claude request: model={request.model}, stream={request.stream}"
        )

        client = registry.get_client_for_model(request.model)
        request_id = str(uuid.uuid4())
        openai_request = convert_claude_to_openai(request, model_mgr)

        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            try:
                openai_stream = client.create_chat_completion_stream(
                    openai_request, request_id
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        client,
                        request_id,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            except HTTPException as e:
                logger.error(f"Streaming error: {e.detail}")
                logger.error(traceback.format_exc())
                error_message = client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            openai_response = await client.create_chat_completion(
                openai_request, request_id
            )
            claude_response = convert_openai_to_claude_response(
                openai_response, request
            )
            return claude_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        try:
            client = registry.get_client_for_model(request.model)
            error_message = client.classify_openai_error(str(e))
        except Exception:
            error_message = str(e)
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: ClaudeTokenCountRequest,
    _auth: None = Depends(validate_api_key),
):
    try:
        total_chars = 0

        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        estimated_tokens = max(1, total_chars // 4)
        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(request: Request):
    config = get_config(request)
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
        "tiers": {
            "opus": {"model": config.opus.model, "base_url": config.opus.base_url, "configured": bool(config.opus.api_key)},
            "sonnet": {"model": config.sonnet.model, "base_url": config.sonnet.base_url, "configured": bool(config.sonnet.api_key)},
            "haiku": {"model": config.haiku.model, "base_url": config.haiku.base_url, "configured": bool(config.haiku.api_key)},
        },
    }


@router.get("/test-connection")
async def test_connection(request: Request, tier: Optional[str] = None):
    registry = get_client_registry(request)
    target_tier = tier or "haiku"
    result = await registry.test_connection(target_tier)
    if result["status"] == "failed":
        return JSONResponse(status_code=503, content=result)
    return result


@router.get("/")
async def root(_auth: None = Depends(validate_api_key)):
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
            "dashboard": "/dashboard",
        },
    }
