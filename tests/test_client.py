"""Unit tests for OpenAI client retry logic."""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import HTTPException
from openai._exceptions import RateLimitError, APIConnectionError, AuthenticationError, BadRequestError


def _make_client(max_retries=2):
    from src.core.client import OpenAIClient
    return OpenAIClient(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        timeout=30,
        max_retries=max_retries,
    )


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        client = _make_client(max_retries=2)
        mock_completion = MagicMock()
        mock_completion.model_dump.return_value = {"id": "test", "choices": []}

        call_count = 0
        async def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(message="rate limited", response=MagicMock(status_code=429), body=None)
            return mock_completion

        client.client.chat.completions.create = flaky_create
        result = await client.create_chat_completion({"model": "gpt-4o", "messages": [], "max_tokens": 10})
        assert call_count == 2
        assert result == {"id": "test", "choices": []}

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        client = _make_client(max_retries=1)
        mock_completion = MagicMock()
        mock_completion.model_dump.return_value = {"id": "test", "choices": []}

        call_count = 0
        async def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIConnectionError(request=MagicMock())
            return mock_completion

        client.client.chat.completions.create = flaky_create
        result = await client.create_chat_completion({"model": "gpt-4o", "messages": [], "max_tokens": 10})
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self):
        client = _make_client(max_retries=3)

        async def fail_auth(**kwargs):
            raise AuthenticationError(message="bad key", response=MagicMock(status_code=401), body=None)

        client.client.chat.completions.create = fail_auth
        with pytest.raises(HTTPException) as exc_info:
            await client.create_chat_completion({"model": "gpt-4o", "messages": [], "max_tokens": 10})
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_no_retry_on_bad_request(self):
        client = _make_client(max_retries=3)

        async def fail_bad_request(**kwargs):
            raise BadRequestError(message="bad request", response=MagicMock(status_code=400), body=None)

        client.client.chat.completions.create = fail_bad_request
        with pytest.raises(HTTPException) as exc_info:
            await client.create_chat_completion({"model": "gpt-4o", "messages": [], "max_tokens": 10})
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        client = _make_client(max_retries=1)

        async def always_fail(**kwargs):
            raise RateLimitError(message="rate limited", response=MagicMock(status_code=429), body=None)

        client.client.chat.completions.create = always_fail
        with pytest.raises(HTTPException) as exc_info:
            await client.create_chat_completion({"model": "gpt-4o", "messages": [], "max_tokens": 10})
        assert exc_info.value.status_code == 429


class TestCancellation:
    def test_cancel_request(self):
        client = _make_client()
        import asyncio
        event = asyncio.Event()
        client.active_requests["req-1"] = event

        result = client.cancel_request("req-1")
        assert result is True
        assert event.is_set()

    def test_cancel_unknown_request(self):
        client = _make_client()
        result = client.cancel_request("nonexistent")
        assert result is False


class TestErrorClassification:
    def test_region_error(self):
        client = _make_client()
        msg = client.classify_openai_error("unsupported_country_region_territory")
        assert "not available in your region" in msg

    def test_invalid_key_error(self):
        client = _make_client()
        msg = client.classify_openai_error("invalid_api_key")
        assert "Invalid API key" in msg

    def test_rate_limit_error(self):
        client = _make_client()
        msg = client.classify_openai_error("rate_limit exceeded")
        assert "Rate limit" in msg

    def test_model_not_found_error(self):
        client = _make_client()
        msg = client.classify_openai_error("model not found")
        assert "Model not found" in msg

    def test_billing_error(self):
        client = _make_client()
        msg = client.classify_openai_error("billing issue detected")
        assert "Billing" in msg

    def test_unknown_error_passthrough(self):
        client = _make_client()
        msg = client.classify_openai_error("something unexpected")
        assert msg == "something unexpected"
