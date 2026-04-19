import hmac
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    api_key: str
    base_url: str
    model: str
    api_version: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)


def _get_custom_headers(prefix: str = "CUSTOM_HEADER_") -> Dict[str, str]:
    custom_headers = {}
    for env_key, env_value in os.environ.items():
        if env_key.startswith(prefix):
            header_name = env_key[len(prefix):]
            if header_name:
                header_name = header_name.replace('_', '-')
                custom_headers[header_name] = env_value
    return custom_headers


class Config:
    def __init__(self):
        # Global defaults
        global_api_key = os.environ.get("OPENAI_API_KEY", "")
        global_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        global_custom_headers = _get_custom_headers()

        # Per-tier provider configs with fallback to global vars
        self.opus = ProviderConfig(
            api_key=os.environ.get("OPUS_API_KEY", global_api_key),
            base_url=os.environ.get("OPUS_BASE_URL", global_base_url),
            model=os.environ.get("BIG_MODEL", "gpt-4o"),
            api_version=os.environ.get("OPUS_API_VERSION", os.environ.get("AZURE_API_VERSION")),
            custom_headers=global_custom_headers.copy(),
        )
        self.sonnet = ProviderConfig(
            api_key=os.environ.get("SONNET_API_KEY", global_api_key),
            base_url=os.environ.get("SONNET_BASE_URL", global_base_url),
            model=os.environ.get("MIDDLE_MODEL", os.environ.get("BIG_MODEL", "gpt-4o")),
            api_version=os.environ.get("SONNET_API_VERSION", os.environ.get("AZURE_API_VERSION")),
            custom_headers=global_custom_headers.copy(),
        )
        self.haiku = ProviderConfig(
            api_key=os.environ.get("HAIKU_API_KEY", global_api_key),
            base_url=os.environ.get("HAIKU_BASE_URL", global_base_url),
            model=os.environ.get("SMALL_MODEL", "gpt-4o-mini"),
            api_version=os.environ.get("HAIKU_API_VERSION", os.environ.get("AZURE_API_VERSION")),
            custom_headers=global_custom_headers.copy(),
        )

        # Backward-compatible aliases
        self.openai_api_key = self.sonnet.api_key
        self.openai_base_url = self.sonnet.base_url
        self.big_model = self.opus.model
        self.middle_model = self.sonnet.model
        self.small_model = self.haiku.model
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")

        # Client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")

        # Server settings
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))

        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # Limits
        self.max_body_mb = int(os.environ.get("MAX_BODY_MB", "10"))
        self.rate_limit = int(os.environ.get("RATE_LIMIT", "60"))

    def get_tier(self, tier: str) -> ProviderConfig:
        tiers = {"opus": self.opus, "sonnet": self.sonnet, "haiku": self.haiku}
        return tiers[tier]

    def update_tier(self, tier: str, **kwargs):
        provider = self.get_tier(tier)
        for key, value in kwargs.items():
            if value is not None and hasattr(provider, key):
                setattr(provider, key, value)
        # Sync backward-compat aliases
        self.big_model = self.opus.model
        self.middle_model = self.sonnet.model
        self.small_model = self.haiku.model

    def validate_api_key(self):
        return bool(self.openai_api_key)

    def validate_client_api_key(self, client_api_key):
        if not self.anthropic_api_key:
            return True
        if not client_api_key or not self.anthropic_api_key:
            return False
        return hmac.compare_digest(client_api_key, self.anthropic_api_key)

    def get_custom_headers(self):
        return _get_custom_headers()

    def to_env_dict(self) -> Dict[str, str]:
        """Serialize config to env var dict for persistence."""
        d = {}
        for tier_name, prefix, provider in [
            ("opus", "OPUS", self.opus),
            ("sonnet", "SONNET", self.sonnet),
            ("haiku", "HAIKU", self.haiku),
        ]:
            d[f"{prefix}_API_KEY"] = provider.api_key
            d[f"{prefix}_BASE_URL"] = provider.base_url
            d[f"{prefix}_MODEL"] = provider.model
            if provider.api_version:
                d[f"{prefix}_API_VERSION"] = provider.api_version
        # Legacy keys for backward compat
        d["BIG_MODEL"] = self.opus.model
        d["MIDDLE_MODEL"] = self.sonnet.model
        d["SMALL_MODEL"] = self.haiku.model
        d["OPENAI_BASE_URL"] = self.openai_base_url
        d["HOST"] = self.host
        d["PORT"] = str(self.port)
        d["LOG_LEVEL"] = self.log_level
        d["MAX_TOKENS_LIMIT"] = str(self.max_tokens_limit)
        d["MIN_TOKENS_LIMIT"] = str(self.min_tokens_limit)
        d["REQUEST_TIMEOUT"] = str(self.request_timeout)
        d["MAX_RETRIES"] = str(self.max_retries)
        if self.anthropic_api_key:
            d["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        return d
