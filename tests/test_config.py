"""Unit tests for request and response conversion."""

import json
import os
import sys
import pytest

from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage, ClaudeTool


class FakeConfig:
    def __init__(self):
        self.big_model = "gpt-4o"
        self.middle_model = "gpt-4o"
        self.small_model = "gpt-4o-mini"
        self.min_tokens_limit = 100
        self.max_tokens_limit = 4096


class FakeModelManager:
    def __init__(self):
        self.config = FakeConfig()

    def map_claude_model_to_openai(self, model):
        if "haiku" in model.lower():
            return "gpt-4o-mini"
        return "gpt-4o"


# ── Request Conversion ──

class TestClaudeToOpenAIConversion:
    def _convert(self, **kwargs):
        from src.conversion.request_converter import convert_claude_to_openai
        defaults = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        defaults.update(kwargs)
        request = ClaudeMessagesRequest(**defaults)
        return convert_claude_to_openai(request, FakeModelManager())

    def test_basic_text_message(self):
        result = self._convert()
        assert result["model"] == "gpt-4o"
        assert result["messages"][-1]["role"] == "user"
        assert result["messages"][-1]["content"] == "Hello"
        assert result["stream"] is False

    def test_system_message_string(self):
        result = self._convert(system="You are helpful")
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"

    def test_system_message_list(self):
        result = self._convert(system=[{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}])
        sys_msg = result["messages"][0]
        assert sys_msg["role"] == "system"
        assert "Part 1" in sys_msg["content"]
        assert "Part 2" in sys_msg["content"]

    def test_tools_conversion(self):
        tools = [
            ClaudeTool(name="get_weather", description="Get weather", input_schema={"type": "object", "properties": {"loc": {"type": "string"}}})
        ]
        result = self._convert(tools=tools)
        assert "tools" in result
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_tool_choice_auto(self):
        result = self._convert(tool_choice={"type": "auto"})
        assert result["tool_choice"] == "auto"

    def test_tool_choice_named(self):
        result = self._convert(tool_choice={"type": "tool", "name": "get_weather"})
        assert result["tool_choice"]["function"]["name"] == "get_weather"

    def test_max_tokens_clamped(self):
        result = self._convert(max_tokens=999999)
        assert result["max_tokens"] == 4096

    def test_max_tokens_floored(self):
        result = self._convert(max_tokens=1)
        assert result["max_tokens"] == 100

    def test_multimodal_image(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}}
            ]
        }]
        result = self._convert(messages=messages)
        user_msg = result["messages"][-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][1]["type"] == "image_url"

    def test_assistant_with_tool_use(self):
        messages = [
            {"role": "user", "content": "Check weather"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"loc": "NYC"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "Sunny, 72F"}
            ]}
        ]
        result = self._convert(messages=messages)
        assistant_msg = [m for m in result["messages"] if m["role"] == "assistant"][0]
        assert assistant_msg["content"] == "Let me check."
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"
        tool_msg = [m for m in result["messages"] if m["role"] == "tool"][0]
        assert tool_msg["content"] == "Sunny, 72F"

    def test_thinking_parameter_ignored(self):
        result = self._convert(thinking={"enabled": True})
        assert "thinking" not in result


# ── Response Conversion ──

class TestOpenAIToClaudeConversion:
    def _convert(self, openai_response, model="claude-3-5-sonnet-20241022"):
        from src.conversion.response_converter import convert_openai_to_claude_response
        request = ClaudeMessagesRequest(model=model, max_tokens=1024, messages=[{"role": "user", "content": "hi"}])
        return convert_openai_to_claude_response(openai_response, request)

    def test_basic_text_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello!", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        result = self._convert(openai_resp)
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"

    def test_tool_call_response(self):
        openai_resp = {
            "id": "chatcmpl-456",
            "choices": [{"message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"loc\": \"NYC\"}"}}]
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10}
        }
        result = self._convert(openai_resp)
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "get_weather"
        assert result["content"][0]["input"] == {"loc": "NYC"}
        assert result["stop_reason"] == "tool_use"

    def test_max_tokens_stop_reason(self):
        openai_resp = {
            "id": "chatcmpl-789",
            "choices": [{"message": {"content": "Truncated", "role": "assistant"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        result = self._convert(openai_resp)
        assert result["stop_reason"] == "max_tokens"

    def test_empty_content_gets_empty_text(self):
        openai_resp = {
            "id": "chatcmpl-000",
            "choices": [{"message": {"content": None, "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0}
        }
        result = self._convert(openai_resp)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == ""

    def test_no_choices_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            self._convert({"id": "x", "choices": [], "usage": {}})
        assert exc_info.value.status_code == 500
