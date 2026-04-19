"""Unit tests for API endpoints using dependency injection."""

import os
import sys
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.dependencies import init_app_state
from src.api.endpoints import router


def _make_app(anthropic_key=None):
    """Create a fresh app with controlled config."""
    from src.core.config import Config

    app = FastAPI()
    cfg = Config()
    cfg.anthropic_api_key = anthropic_key
    init_app_state(app, config=cfg)
    app.include_router(router)
    return app


@pytest.fixture
def no_auth_client():
    return TestClient(_make_app(anthropic_key=None), raise_server_exceptions=False)


@pytest.fixture
def auth_client():
    return TestClient(_make_app(anthropic_key="expected-key"), raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_200(self, no_auth_client):
        resp = no_auth_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_shows_api_key_configured(self, no_auth_client):
        resp = no_auth_client.get("/health")
        assert resp.json()["openai_api_configured"] is True


class TestRootEndpoint:
    def test_root_requires_auth_without_key(self, auth_client):
        resp = auth_client.get("/")
        assert resp.status_code == 401

    def test_root_allows_with_valid_key(self, auth_client):
        resp = auth_client.get("/", headers={"x-api-key": "expected-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "config" not in data
        assert "endpoints" in data

    def test_root_rejects_bad_key(self, auth_client):
        resp = auth_client.get("/", headers={"x-api-key": "wrong-key"})
        assert resp.status_code == 401

    def test_root_accepts_bearer_auth(self, auth_client):
        resp = auth_client.get("/", headers={"Authorization": "Bearer expected-key"})
        assert resp.status_code == 200

    def test_root_allows_without_validation_when_no_key_set(self, no_auth_client):
        resp = no_auth_client.get("/")
        assert resp.status_code == 200


class TestMessagesEndpoint:
    def test_messages_requires_auth(self, auth_client):
        resp = auth_client.post("/v1/messages", json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert resp.status_code == 401


class TestCountTokens:
    def test_count_tokens_basic(self, no_auth_client):
        resp = no_auth_client.post("/v1/messages/count_tokens", json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello world"}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "input_tokens" in data
        assert data["input_tokens"] > 0

    def test_count_tokens_with_system(self, no_auth_client):
        resp = no_auth_client.post("/v1/messages/count_tokens", json={
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] > 1
