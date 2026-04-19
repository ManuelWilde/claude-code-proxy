import json
import uuid
from fastapi import HTTPException, Request
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest


def _sse_event(event: str, data: dict) -> str:
    """Format an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Add text content
    text_content = message.get("content")
    if text_content is not None:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
        },
    }

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request = None,
    openai_client=None,
    request_id: str = None,
):
    """Convert OpenAI streaming response to Claude streaming format.

    Supports optional cancellation when http_request, openai_client, and
    request_id are provided.
    """

    has_cancellation = http_request is not None and openai_client is not None and request_id is not None

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield _sse_event(Constants.EVENT_MESSAGE_START, {
        "type": Constants.EVENT_MESSAGE_START,
        "message": {
            "id": message_id,
            "type": "message",
            "role": Constants.ROLE_ASSISTANT,
            "model": original_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    yield _sse_event(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    # Process streaming chunks — text block is opened lazily on first text delta
    text_block_opened = False
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = {"input_tokens": 0, "output_tokens": 0}

    try:
        async for line in openai_stream:
            # Check if client disconnected (cancellation support)
            if has_cancellation and await http_request.is_disconnected():
                logger.info(f"Client disconnected, cancelling request {request_id}")
                openai_client.cancel_request(request_id)
                break

            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        usage = chunk.get("usage")
                        if usage:
                            prompt_tokens_details = usage.get('prompt_tokens_details', {})
                            cached_tokens = prompt_tokens_details.get('cached_tokens', 0) if prompt_tokens_details else 0
                            usage_data = {
                                'input_tokens': usage.get('prompt_tokens', 0),
                                'output_tokens': usage.get('completion_tokens', 0),
                                'cache_read_input_tokens': cached_tokens,
                            }
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle text delta — open text block lazily
                    if delta and "content" in delta and delta["content"] is not None:
                        if not text_block_opened:
                            text_block_opened = True
                            yield _sse_event(Constants.EVENT_CONTENT_BLOCK_START, {
                                "type": Constants.EVENT_CONTENT_BLOCK_START,
                                "index": text_block_index,
                                "content_block": {"type": Constants.CONTENT_TEXT, "text": ""},
                            })
                        yield _sse_event(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                            "index": text_block_index,
                            "delta": {"type": Constants.DELTA_TEXT, "text": delta["content"]},
                        })

                    # Handle tool call deltas with incremental processing
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "claude_index": None,
                                    "started": False,
                                }

                            tool_call = current_tool_calls[tc_index]

                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]

                            # Start content block when we have both id and name
                            if tool_call["id"] and tool_call["name"] and not tool_call["started"]:
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True

                                yield _sse_event(Constants.EVENT_CONTENT_BLOCK_START, {
                                    "type": Constants.EVENT_CONTENT_BLOCK_START,
                                    "index": claude_index,
                                    "content_block": {
                                        "type": Constants.CONTENT_TOOL_USE,
                                        "id": tool_call["id"],
                                        "name": tool_call["name"],
                                        "input": {},
                                    },
                                })

                            # Send incremental argument deltas
                            if "arguments" in function_data and tool_call["started"] and function_data["arguments"] is not None:
                                new_chunk = function_data["arguments"]
                                tool_call["args_buffer"] += new_chunk

                                if new_chunk:
                                    yield _sse_event(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                                        "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                                        "index": tool_call["claude_index"],
                                        "delta": {
                                            "type": Constants.DELTA_INPUT_JSON,
                                            "partial_json": new_chunk,
                                        },
                                    })

                    # Handle finish reason
                    if finish_reason:
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        else:
                            final_stop_reason = Constants.STOP_END_TURN

    except HTTPException as e:
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            yield _sse_event("error", {
                "type": "error",
                "error": {"type": "cancelled", "message": "Request was cancelled by client"},
            })
            return
        else:
            raise
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        yield _sse_event("error", {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        })
        return

    # Close text block if it was opened
    if text_block_opened:
        yield _sse_event(Constants.EVENT_CONTENT_BLOCK_STOP, {
            "type": Constants.EVENT_CONTENT_BLOCK_STOP,
            "index": text_block_index,
        })

    # Close tool blocks
    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield _sse_event(Constants.EVENT_CONTENT_BLOCK_STOP, {
                "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                "index": tool_data["claude_index"],
            })

    yield _sse_event(Constants.EVENT_MESSAGE_DELTA, {
        "type": Constants.EVENT_MESSAGE_DELTA,
        "delta": {"stop_reason": final_stop_reason, "stop_sequence": None},
        "usage": usage_data,
    })

    yield _sse_event(Constants.EVENT_MESSAGE_STOP, {
        "type": Constants.EVENT_MESSAGE_STOP,
    })
