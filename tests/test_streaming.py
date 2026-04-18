"""Unit tests for streaming response conversion."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest


class _AsyncIter:
    """Wrap a sync iterable for async for loops."""
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _make_request():
    return ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "hi"}],
    )


async def _collect_events(stream):
    """Collect all SSE event strings from an async generator."""
    events = []
    async for event in stream:
        events.append(event)
    return events


def _parse_sse(event_str):
    """Parse an SSE string into (event_name, data_dict)."""
    lines = event_str.strip().split("\n")
    event_name = lines[0].split(": ", 1)[1]
    data = json.loads(lines[1].split(": ", 1)[1])
    return event_name, data


class TestStreamingTextOnly:
    @pytest.mark.asyncio
    async def test_text_streaming_events(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)

        event_types = [_parse_sse(e)[0] for e in events]
        assert event_types[0] == Constants.EVENT_MESSAGE_START
        assert event_types[1] == Constants.EVENT_CONTENT_BLOCK_START
        assert event_types[2] == Constants.EVENT_PING
        assert Constants.EVENT_CONTENT_BLOCK_DELTA in event_types
        assert Constants.EVENT_CONTENT_BLOCK_STOP in event_types
        assert Constants.EVENT_MESSAGE_DELTA in event_types
        assert event_types[-1] == Constants.EVENT_MESSAGE_STOP

    @pytest.mark.asyncio
    async def test_text_delta_content(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = [
            'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"!"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)

        text_deltas = []
        for e in events:
            name, data = _parse_sse(e)
            if name == Constants.EVENT_CONTENT_BLOCK_DELTA:
                if data.get("delta", {}).get("type") == Constants.DELTA_TEXT:
                    text_deltas.append(data["delta"]["text"])

        assert text_deltas == ["Hi", "!"]


class TestStreamingToolCalls:
    @pytest.mark.asyncio
    async def test_tool_call_streaming(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = [
            'data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"function":{"arguments":"{\\"loc\\""}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"function":{"arguments":": \\"NYC\\"}"}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)

        event_types = [_parse_sse(e)[0] for e in events]
        assert Constants.EVENT_CONTENT_BLOCK_START in event_types
        assert Constants.EVENT_CONTENT_BLOCK_STOP in event_types

        # Find tool_use content block start
        tool_start = None
        for e in events:
            name, data = _parse_sse(e)
            if name == Constants.EVENT_CONTENT_BLOCK_START:
                block = data.get("content_block", {})
                if block.get("type") == Constants.CONTENT_TOOL_USE:
                    tool_start = block
        assert tool_start is not None
        assert tool_start["name"] == "get_weather"
        assert tool_start["id"] == "call_1"

    @pytest.mark.asyncio
    async def test_tool_call_stop_reason(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = [
            'data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"test","arguments":""}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)

        for e in events:
            name, data = _parse_sse(e)
            if name == Constants.EVENT_MESSAGE_DELTA:
                assert data["delta"]["stop_reason"] == Constants.STOP_TOOL_USE


class TestStreamingEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_stream(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = ["data: [DONE]"]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)

        event_types = [_parse_sse(e)[0] for e in events]
        assert Constants.EVENT_MESSAGE_START in event_types
        assert Constants.EVENT_MESSAGE_STOP in event_types

    @pytest.mark.asyncio
    async def test_max_tokens_stop(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = [
            'data: {"choices":[{"delta":{"content":"Truncated text"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
            "data: [DONE]",
        ]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)

        for e in events:
            name, data = _parse_sse(e)
            if name == Constants.EVENT_MESSAGE_DELTA:
                assert data["delta"]["stop_reason"] == Constants.STOP_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_usage_data_extracted(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = [
            'data: {"usage":{"prompt_tokens":15,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":5}},"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)

        for e in events:
            name, data = _parse_sse(e)
            if name == Constants.EVENT_MESSAGE_DELTA:
                assert data["usage"]["input_tokens"] == 15
                assert data["usage"]["output_tokens"] == 7
                assert data["usage"]["cache_read_input_tokens"] == 5

    @pytest.mark.asyncio
    async def test_malformed_chunk_skipped(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = [
            'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":null}]}',
            "data: {invalid json}",
            'data: {"choices":[{"delta":{"content":" still going"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)

        # Should still have normal events despite the bad chunk
        text_deltas = []
        for e in events:
            name, data = _parse_sse(e)
            if name == Constants.EVENT_CONTENT_BLOCK_DELTA:
                if data.get("delta", {}).get("type") == Constants.DELTA_TEXT:
                    text_deltas.append(data["delta"]["text"])
        assert "OK" in text_deltas
        assert " still going" in text_deltas

    @pytest.mark.asyncio
    async def test_no_choices_chunk_skipped(self):
        from src.conversion.response_converter import convert_openai_streaming_to_claude
        import logging
        logger = logging.getLogger("test")

        openai_chunks = [
            'data: {"choices":[]}',
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        stream = convert_openai_streaming_to_claude(
            _AsyncIter(openai_chunks), _make_request(), logger
        )
        events = await _collect_events(stream)
        assert len(events) > 0
