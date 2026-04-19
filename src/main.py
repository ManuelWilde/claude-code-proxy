import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.endpoints import router as api_router
from src.api.dashboard import router as dashboard_router
from src.core.config import Config
from src.core.dependencies import init_app_state
from src.core.middleware import RateLimitMiddleware, RequestSizeLimitMiddleware

load_dotenv()

logger = logging.getLogger(__name__)


# Build config once at module scope so middleware values are available
# before the app starts. Starlette forbids add_middleware() after startup.
_boot_config = Config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Configuration loaded: API_KEY=%s, BASE_URL='%s'",
        "set" if _boot_config.openai_api_key else "not set",
        _boot_config.openai_base_url,
    )
    logger.info("  Opus:  %s @ %s", _boot_config.opus.model, _boot_config.opus.base_url)
    logger.info("  Sonnet: %s @ %s", _boot_config.sonnet.model, _boot_config.sonnet.base_url)
    logger.info("  Haiku:  %s @ %s", _boot_config.haiku.model, _boot_config.haiku.base_url)

    init_app_state(app, _boot_config)

    yield

    logger.info("Shutting down: closing clients")
    registry = app.state.client_registry
    if registry:
        for tier, client in registry._clients.items():
            if hasattr(client, "client"):
                await client.client.close()
    logger.info("Shutdown complete")


app = FastAPI(title="Claude-to-OpenAI API Proxy", version="1.0.0", lifespan=lifespan)

# Middleware MUST be registered before the app starts serving; do it here at
# module scope rather than inside lifespan (Starlette >= 0.35 rejects late adds).
app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=_boot_config.max_body_mb * 1024 * 1024)
app.add_middleware(RateLimitMiddleware, max_requests=_boot_config.rate_limit, window_seconds=60)

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS origins configured: %s", _cors_origins)

app.include_router(api_router)
app.include_router(dashboard_router)


def main():
    import sys

    # Load config for CLI help / startup messages
    config = Config()

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Claude-to-OpenAI API Proxy v1.0.0")
        print("")
        print("Usage: python src/main.py")
        print("")
        print("Environment variables:")
        print("  OPENAI_API_KEY - Default API key for all tiers")
        print("  OPUS_API_KEY   - API key for opus tier (overrides OPENAI_API_KEY)")
        print("  SONNET_API_KEY - API key for sonnet tier (overrides OPENAI_API_KEY)")
        print("  HAIKU_API_KEY  - API key for haiku tier (overrides OPENAI_API_KEY)")
        print("")
        print("  OPUS_BASE_URL / SONNET_BASE_URL / HAIKU_BASE_URL - Per-tier base URLs")
        print("  OPUS_MODEL / SONNET_MODEL / HAIKU_MODEL - Per-tier model names")
        print("  BIG_MODEL / MIDDLE_MODEL / SMALL_MODEL - Legacy model name aliases")
        print("  OPENAI_BASE_URL - Default base URL for all tiers")
        print("  ANTHROPIC_API_KEY - Expected client API key for validation")
        print(f"  HOST - Server host (default: 0.0.0.0)")
        print(f"  PORT - Server port (default: 8082)")
        print(f"  LOG_LEVEL - Logging level (default: INFO)")
        print("")
        print(f"  Dashboard: http://{config.host}:{config.port}/dashboard")
        sys.exit(0)

    print("Claude-to-OpenAI API Proxy v1.0.0")
    print(f"  Opus:  {config.opus.model} @ {config.opus.base_url}")
    print(f"  Sonnet: {config.sonnet.model} @ {config.sonnet.base_url}")
    print(f"  Haiku:  {config.haiku.model} @ {config.haiku.base_url}")
    print(f"  Server: {config.host}:{config.port}")
    print(f"  Dashboard: http://localhost:{config.port}/dashboard")
    print(f"  API Key Validation: {'Enabled' if config.anthropic_api_key else 'Disabled'}")
    dashboard_pw = os.environ.get("DASHBOARD_PASSWORD") or os.environ.get("ANTHROPIC_API_KEY")
    if dashboard_pw:
        print(f"  Dashboard Auth: Uses {'DASHBOARD_PASSWORD' if os.environ.get('DASHBOARD_PASSWORD') else 'ANTHROPIC_API_KEY'}")
    else:
        print("  Dashboard Auth: Auto-generated token (check logs for details)")
    print("")

    log_level = config.log_level.split()[0].lower()
    valid_levels = ["debug", "info", "warning", "error", "critical"]
    if log_level not in valid_levels:
        log_level = "info"

    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
