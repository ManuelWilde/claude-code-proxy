import logging
from datetime import datetime
from typing import Dict, Optional

from src.core.client import OpenAIClient
from src.core.config import Config
from src.core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ClientRegistry:
    """Manages OpenAIClient instances per model tier (opus/sonnet/haiku)."""

    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self._clients: Dict[str, OpenAIClient] = {}
        self._build_clients()

    def _build_clients(self):
        """Create OpenAIClient for each tier that has an API key."""
        self._clients = {}
        for tier_name in ("opus", "sonnet", "haiku"):
            provider = self.config.get_tier(tier_name)
            if provider.api_key:
                self._clients[tier_name] = OpenAIClient(
                    api_key=provider.api_key,
                    base_url=provider.base_url,
                    timeout=self.config.request_timeout,
                    api_version=provider.api_version,
                    custom_headers=provider.custom_headers,
                    max_retries=self.config.max_retries,
                )
                logger.info(f"Client for tier '{tier_name}': {provider.model} @ {provider.base_url}")
            else:
                logger.warning(f"No API key for tier '{tier_name}', client not created")

    def refresh(self, config: Optional[Config] = None):
        """Rebuild all clients (called after config update)."""
        if config:
            self.config = config
            self.model_manager = ModelManager(config)
        self._build_clients()
        logger.info("Client registry refreshed")

    def get_client_for_model(self, claude_model: str) -> OpenAIClient:
        """Get the appropriate client for a Claude model name."""
        tier = self.model_manager.get_tier(claude_model)
        if tier in self._clients:
            return self._clients[tier]
        # Fallback: try any available client
        if self._clients:
            fallback = next(iter(self._clients.values()))
            logger.warning(f"No client for tier '{tier}', using fallback")
            return fallback
        raise RuntimeError("No API clients configured. Please configure at least one provider.")

    async def test_connection(self, tier: str) -> Dict:
        """Test connectivity for a specific tier."""
        if tier not in self._clients:
            return {
                "status": "not_configured",
                "message": f"No API key configured for {tier}",
                "timestamp": datetime.now().isoformat(),
            }

        provider = self.config.get_tier(tier)
        client = self._clients[tier]
        try:
            response = await client.create_chat_completion({
                "model": provider.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            })
            return {
                "status": "success",
                "message": f"Connected successfully via {provider.model}",
                "model": provider.model,
                "base_url": provider.base_url,
                "timestamp": datetime.now().isoformat(),
                "response_id": response.get("id", "unknown"),
            }
        except Exception as e:
            return {
                "status": "failed",
                "message": str(e),
                "model": provider.model,
                "base_url": provider.base_url,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_all_status(self) -> Dict:
        """Get connectivity status for all tiers."""
        results = {}
        for tier_name in ("opus", "sonnet", "haiku"):
            results[tier_name] = await self.test_connection(tier_name)
        return results
