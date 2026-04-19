import json
import logging
from typing import Dict, Any, List
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage

logger = logging.getLogger(__name__)


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Warn about unsupported thinking/extended thinking parameter
    if claude_request.thinking and claude_request.thinking.enabled:
        logger.warning("Extended thinking is not supported by OpenAI-compatible providers; ignoring thinking parameter")

    # Convert messages
    openai_messages = []

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            openai_messages.append(openai_message)

            if _next_message_has_tool_results(claude_request.messages, i):
                i += 1  # Skip to tool result message
                tool_results, extra_text = _extract_tool_results_with_text(
                    claude_request.messages[i]
                )
                openai_messages.extend(tool_results)
                if extra_text:
                    openai_messages.append(
                        {"role": Constants.ROLE_USER, "content": extra_text}
                    )

        i += 1

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": min(
            max(claude_request.max_tokens, model_manager.config.min_tokens_limit),
            model_manager.config.max_tokens_limit,
        ),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    logger.debug(
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    return openai_request


def _next_message_has_tool_results(messages: List[ClaudeMessage], current_index: int) -> bool:
    """Check if the next message contains tool results."""
    if current_index + 1 >= len(messages):
        return False
    next_msg = messages[current_index + 1]
    if next_msg.role != Constants.ROLE_USER or not isinstance(next_msg.content, list):
        return False
    return any(
        getattr(block, "type", None) == Constants.CONTENT_TOOL_RESULT
        for block in next_msg.content
    )


def _extract_tool_results_with_text(
    msg: ClaudeMessage,
) -> tuple:
    """Extract tool results and any interleaved text from a mixed message."""
    tool_messages = []
    text_parts = []

    if isinstance(msg.content, list):
        for block in msg.content:
            block_type = getattr(block, "type", None)
            if block_type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(
                    getattr(block, "content", None)
                )
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": getattr(block, "tool_use_id", ""),
                        "content": content,
                    }
                )
            elif block_type == Constants.CONTENT_TEXT:
                text = getattr(block, "text", "")
                if text:
                    text_parts.append(text)

    extra_text = "\n".join(text_parts).strip() if text_parts else None
    return tool_messages, extra_text


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    openai_content = []
    for block in msg.content:
        block_type = getattr(block, "type", None)
        if block_type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": getattr(block, "text", "")})
        elif block_type == Constants.CONTENT_IMAGE:
            source = getattr(block, "source", None)
            if (
                isinstance(source, dict)
                and source.get("type") == "base64"
                and "media_type" in source
                and "data" in source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source['media_type']};base64,{source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        return {"role": Constants.ROLE_ASSISTANT, "content": None}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        block_type = getattr(block, "type", None)
        if block_type == Constants.CONTENT_TEXT:
            text_parts.append(getattr(block, "text", ""))
        elif block_type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": getattr(block, "id", ""),
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": getattr(block, "name", ""),
                        "arguments": json.dumps(getattr(block, "input", {}), ensure_ascii=False),
                    },
                }
            )

    openai_message = {"role": Constants.ROLE_ASSISTANT}
    openai_message["content"] = "".join(text_parts) if text_parts else None

    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return _join_content_items(content)

    if isinstance(content, dict):
        return _content_item_to_str(content)

    try:
        return str(content)
    except Exception:
        return "Unparseable content"


def _content_item_to_str(item) -> str:
    """Convert a single content item to string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if item.get("type") == Constants.CONTENT_TEXT or "text" in item:
            return item.get("text", "")
        try:
            return json.dumps(item, ensure_ascii=False)
        except Exception:
            return str(item)
    return str(item)


def _join_content_items(items: list) -> str:
    """Join a list of mixed content items into a single string."""
    return "\n".join(_content_item_to_str(item) for item in items).strip()
