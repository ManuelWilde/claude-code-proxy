"""FastAPI dependency injection — config, client_registry, model_manager are attachable to app.state."""

import logging
from typing import Optional

from fastapi import FastAPI, Request

from src.core.config import Config
from src.core.client_registry import ClientRegistry
from src.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# Module-level reference for access outside request context
_registry: Optional[ClientRegistry] = None


def init_app_state(app: FastAPI, config: Config) -> None:
    """Wire up app.state with config, client_registry, and model_manager."""
    global _registry

    app.state.config = config
    app.state.model_manager = ModelManager(config)
    registry = ClientRegistry(config)
    app.state.client_registry = registry
    _registry = registry


def get_config(request: Request) -> Config:
    return request.app.state.config


def get_client_registry(request: Request) -> ClientRegistry:
    return request.app.state.client_registry


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


def get_registry() -> Optional[ClientRegistry]:
    """Get the client registry outside request context (e.g., for test-connection)."""
    return _registry
