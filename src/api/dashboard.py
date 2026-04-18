import os
import re
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.core.config import Config
from src.core.client_registry import ClientRegistry
from src.core.dependencies import get_config, get_client_registry
from src.core.env_persistence import update_env, get_env_path
from src.core.logging import logger
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import convert_openai_to_claude_response
from src.models.claude import ClaudeMessagesRequest

router = APIRouter()


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "****" if key else ""
    return f"****{key[-4:]}"


def _validate_api_key(key: str) -> bool:
    """Basic API key validation: non-empty, reasonable length, no whitespace."""
    if not key or not key.strip():
        return False
    if len(key) > 512:
        return False
    if re.search(r'\s', key):
        return False
    return True


def _validate_base_url(url: str) -> bool:
    """Basic URL validation: must start with http:// or https://."""
    if not url or not url.strip():
        return False
    return url.strip().startswith(("http://", "https://"))


# Known provider templates
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "icon": "O",
        "color": "#10a37f",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "o1", "o1-mini", "o3-mini"],
    },
    "azure": {
        "name": "Azure OpenAI",
        "base_url": "https://{resource}.openai.azure.com/openai/deployments/{deployment}",
        "icon": "A",
        "color": "#0078d4",
        "needs_version": True,
        "models": ["gpt-4o", "gpt-4", "gpt-35-turbo"],
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "icon": "G",
        "color": "#f55036",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "icon": "D",
        "color": "#4d6bfe",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "icon": "R",
        "color": "#6d28d9",
        "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-70b-instruct", "google/gemini-pro-1.5"],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1",
        "icon": "L",
        "color": "#6366f1",
        "models": ["llama3.1:70b", "llama3.1:8b", "codellama:34b", "mistral:7b"],
    },
    "together": {
        "name": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "icon": "T",
        "color": "#3b82f6",
        "models": ["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
    },
    "custom": {
        "name": "Custom Provider",
        "base_url": "",
        "icon": "?",
        "color": "#6b7280",
        "models": [],
    },
}


def _detect_provider(base_url: str) -> str:
    """Auto-detect provider from base URL using hostname matching."""
    from urllib.parse import urlparse

    url = base_url.lower().rstrip("/")
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if "azure" in hostname:
        return "azure"
    if hostname in ("localhost", "127.0.0.1") and parsed.port == 11434:
        return "ollama"

    for key, info in PROVIDERS.items():
        if key in ("azure", "ollama", "custom"):
            continue
        template_parsed = urlparse(info["base_url"].lower().rstrip("/"))
        if template_parsed.hostname and hostname == template_parsed.hostname:
            return key

    return "custom"


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    html_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


@router.get("/api/config")
async def get_config_endpoint(request: Request):
    """Get current config (API keys masked)."""
    config = get_config(request)
    tiers = {}
    for name in ("opus", "sonnet", "haiku"):
        p = config.get_tier(name)
        tiers[name] = {
            "api_key_set": bool(p.api_key),
            "api_key_preview": _mask_key(p.api_key),
            "base_url": p.base_url,
            "model": p.model,
            "api_version": p.api_version,
            "provider": _detect_provider(p.base_url),
        }
    return {
        "tiers": tiers,
        "server": {
            "host": config.host,
            "port": config.port,
            "log_level": config.log_level,
        },
        "performance": {
            "max_tokens_limit": config.max_tokens_limit,
            "min_tokens_limit": config.min_tokens_limit,
            "request_timeout": config.request_timeout,
            "max_retries": config.max_retries,
        },
    }


@router.put("/api/config")
async def update_config_endpoint(request: Request):
    """Update config from dashboard, persist to .env, refresh clients."""
    body = await request.json()
    config = get_config(request)
    registry = get_client_registry(request)
    updates = {}

    for tier_name in ("opus", "sonnet", "haiku"):
        if tier_name not in body.get("tiers", {}):
            continue
        tier_data = body["tiers"][tier_name]
        provider = config.get_tier(tier_name)
        prefix = tier_name.upper()

        if "api_key" in tier_data and tier_data["api_key"]:
            if not _validate_api_key(tier_data["api_key"]):
                return JSONResponse(status_code=400, content={"error": f"Invalid API key for {tier_name}"})
            provider.api_key = tier_data["api_key"]
            updates[f"{prefix}_API_KEY"] = tier_data["api_key"]
            os.environ[f"{prefix}_API_KEY"] = tier_data["api_key"]
        if "base_url" in tier_data:
            if not _validate_base_url(tier_data["base_url"]):
                return JSONResponse(status_code=400, content={"error": f"Invalid base URL for {tier_name}"})
            provider.base_url = tier_data["base_url"]
            updates[f"{prefix}_BASE_URL"] = tier_data["base_url"]
            os.environ[f"{prefix}_BASE_URL"] = tier_data["base_url"]
        if "model" in tier_data:
            model = tier_data["model"].strip()
            if not model or '\n' in model or '\r' in model:
                return JSONResponse(status_code=400, content={"error": f"Invalid model name for {tier_name}"})
            provider.model = model
            updates[f"{prefix}_MODEL"] = model
        if "api_version" in tier_data:
            provider.api_version = tier_data["api_version"] or None
            if tier_data["api_version"]:
                updates[f"{prefix}_API_VERSION"] = tier_data["api_version"]

    # Sync aliases
    config.big_model = config.opus.model
    config.middle_model = config.sonnet.model
    config.small_model = config.haiku.model
    config.openai_api_key = config.sonnet.api_key
    config.openai_base_url = config.sonnet.base_url

    if "performance" in body:
        perf = body["performance"]
        bounds = {
            "max_tokens_limit": (1, 100000),
            "min_tokens_limit": (1, 100000),
            "request_timeout": (1, 600),
            "max_retries": (0, 10),
        }
        parsed = {}
        for key in bounds:
            if key in perf:
                try:
                    val = int(perf[key])
                except (ValueError, TypeError):
                    return JSONResponse(status_code=400, content={"error": f"Invalid value for {key}"})
                lo, hi = bounds[key]
                if val < lo or val > hi:
                    return JSONResponse(status_code=400, content={"error": f"{key} must be between {lo} and {hi}"})
                parsed[key] = val

        # Cross-field validation
        min_t = parsed.get("min_tokens_limit", config.min_tokens_limit)
        max_t = parsed.get("max_tokens_limit", config.max_tokens_limit)
        if min_t > max_t:
            return JSONResponse(status_code=400, content={"error": "min_tokens_limit cannot exceed max_tokens_limit"})

        for key, val in parsed.items():
            setattr(config, key, val)
            updates[key.upper()] = str(val)

    env_path = get_env_path()
    update_env(env_path, updates)
    registry.refresh(config)
    return {"status": "ok", "message": "Configuration updated and saved"}


@router.get("/api/providers")
async def get_providers():
    """Get provider templates with known models."""
    return PROVIDERS


@router.post("/api/discover-models/{tier}")
async def discover_models(tier: str, request: Request):
    """Try to fetch available models from the provider's /v1/models endpoint."""
    if tier not in ("opus", "sonnet", "haiku"):
        return JSONResponse(status_code=400, content={"error": f"Invalid tier: {tier}"})

    config = get_config(request)
    provider = config.get_tier(tier)

    if not provider.api_key:
        return {"models": [], "error": "No API key set for this tier"}

    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Try /v1/models relative to base_url
            models_url = provider.base_url.rstrip("/")
            if models_url.endswith("/v1"):
                models_url = f"{models_url}/models"
            elif not models_url.endswith("/models"):
                models_url = f"{models_url}/models"

            resp = await client.get(models_url, headers=headers)
            if resp.status_code != 200:
                return {"models": [], "error": f"API returned {resp.status_code}"}

            data = resp.json()
            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if model_id:
                    models.append(model_id)
            models.sort()
            return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.post("/api/test/{tier}")
async def test_tier_connection(tier: str, request: Request):
    """Test connection for a specific tier."""
    registry = get_client_registry(request)
    if tier not in ("opus", "sonnet", "haiku"):
        return JSONResponse(status_code=400, content={"error": f"Invalid tier: {tier}"})
    result = await registry.test_connection(tier)
    return result


@router.get("/api/status")
async def get_status(request: Request):
    """Get health status for all tiers."""
    registry = get_client_registry(request)
    return await registry.get_all_status()


@router.post("/api/playground")
async def playground(request: Request):
    """Send a test message through the proxy pipeline."""
    body = await request.json()
    message = body.get("message", "")
    tier = body.get("tier", "sonnet")

    if not message:
        return JSONResponse(status_code=400, content={"error": "Message is required"})

    config = get_config(request)
    registry = get_client_registry(request)

    model_map = {"opus": "claude-3-opus", "sonnet": "claude-3-5-sonnet", "haiku": "claude-3-5-haiku"}
    claude_model = model_map.get(tier, "claude-3-5-sonnet")

    claude_request = ClaudeMessagesRequest(
        model=claude_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": message}],
        stream=False,
    )

    try:
        openai_request = convert_claude_to_openai(claude_request, registry.model_manager)
        client = registry.get_client_for_model(claude_model)
        openai_response = await client.create_chat_completion(openai_request)
        claude_response = convert_openai_to_claude_response(openai_response, claude_request)
        return claude_response
    except Exception as e:
        logger.error(f"Playground error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
