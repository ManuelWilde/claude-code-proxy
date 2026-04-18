import logging

from fastapi import FastAPI
from src.api.endpoints import router as api_router
from src.api.dashboard import router as dashboard_router
from src.core.dependencies import init_app_state
from src.core.middleware import RateLimitMiddleware, RequestSizeLimitMiddleware
import uvicorn
import sys
from src.core.config import config

shutdown_logger = logging.getLogger("shutdown")

app = FastAPI(title="Claude-to-OpenAI API Proxy", version="1.0.0")

app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=config.max_body_mb * 1024 * 1024)
app.add_middleware(RateLimitMiddleware, max_requests=config.rate_limit, window_seconds=60)

init_app_state(app)

app.include_router(api_router)
app.include_router(dashboard_router)


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close OpenAI clients on shutdown."""
    shutdown_logger.info("Shutting down: closing clients")
    registry = app.state.client_registry
    if registry:
        for tier, client in registry._clients.items():
            if hasattr(client, "client"):
                await client.client.close()
    shutdown_logger.info("Shutdown complete")


def main():
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
    print(f"  Configuration loaded")
    print(f"  Opus:  {config.opus.model} @ {config.opus.base_url}")
    print(f"  Sonnet: {config.sonnet.model} @ {config.sonnet.base_url}")
    print(f"  Haiku:  {config.haiku.model} @ {config.haiku.base_url}")
    print(f"  Server: {config.host}:{config.port}")
    print(f"  Dashboard: http://localhost:{config.port}/dashboard")
    print(f"  API Key Validation: {'Enabled' if config.anthropic_api_key else 'Disabled'}")
    print("")

    log_level = config.log_level.split()[0].lower()
    valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
    if log_level not in valid_levels:
        log_level = 'info'

    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
