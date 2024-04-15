import logging
import re
from typing import List, Any, Callable

from PIL.Image import Image

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.llm.common.message import ChatMessage, ToolResultMessage, AssistantMessage, ToolCallMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.util.image import base64_encode_image, blob_as_images
from bpm_ai_core.util.json_schema import sanitize_schema, desanitize_json
from bpm_ai_core.util.linguistics import replace_diacritics

logger = logging.getLogger(__name__)

ACCEPTED_IMAGE_FORMATS = ["jpeg", "png", "gif", "webp"]


async def messages_to_anthropic_dicts(messages: List[ChatMessage]):
    return [await message_to_anthropic_dict(m) for m in messages]


async def message_to_anthropic_dict(message: ChatMessage) -> dict:
    if isinstance(message, AssistantMessage) and message.has_tool_calls():
        return tool_calls_message_to_anthropic_dict(message)
    elif isinstance(message, ToolResultMessage):
        return tool_result_message_to_anthropic_dict(message)
    elif isinstance(message.content, str):
        content = message.content
    elif isinstance(message.content, list):
        content = []
        for e in message.content:
            if isinstance(e, str):
                content.append(str_to_anthropic_text_dict(e))
            elif isinstance(e, Blob) and (e.is_image() or e.is_pdf()):
                images = await blob_as_images(e, accept_formats=ACCEPTED_IMAGE_FORMATS)
                for i, image in enumerate(images):
                    content.append(str_to_anthropic_text_dict(f"Image / Page {i+1}:"))
                    content.append(image_to_anthropic_image_dict(image))
            elif isinstance(e, Blob) and (e.is_text()):
                text = (await e.as_bytes()).decode("utf-8")
                filename = (" name='" + e.path + "'") if e.path else ''
                text = f"<file{filename}>\n{text}\n</file>"
                content.append(str_to_anthropic_text_dict(text))
            else:
                raise ValueError(
                    "Elements in ChatMessage.content must be str or Blob (image/pdf/text)"
                )
    else:
        content = None
        logger.warning(
            "ChatMessage.content must be of type str or List[Union[str, PIL.Image]] if used for chat completions."
        )
    return {
        "role": message.role,
        **({"content": content} if content else {}),
        **({"name": message.name} if message.name else {})
    }


def tool_calls_message_to_anthropic_dict(message: AssistantMessage) -> dict:
    return {
        "role": "assistant",
        "content": ([{"type": "text", "text": message.content}] if message.content else [])
                 + [{"type": "tool_use", "id": call.id, "name": call.name, "input": call.payload}
                    for call in message.tool_calls]
    }


def tool_result_message_to_anthropic_dict(message: ToolResultMessage, is_error: bool = False) -> dict:
    return {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": message.id,
          "content": message.content,
          **({"is_error": True} if is_error else {})
        }
      ]
    }


def image_to_anthropic_image_dict(image: Image) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": f"image/{image.format.lower()}",
            "data": base64_encode_image(image),
        }
    }


def str_to_anthropic_text_dict(text: str) -> dict:
    return {
        "type": "text",
        "text": text
    }


def json_schema_to_anthropic_tool(name: str, desc: str, schema: dict[str, Any], key_mapping: dict) -> dict:
    return {
        "name": name,
        "description": desc,
        "input_schema": sanitize_schema(schema, sanitize_key, key_mapping)
    }


def sanitize_key(item):
    return re.sub(r'[^a-zA-Z0-9_-]+', '', replace_diacritics(item).replace(' ', '_'))[:64]


def tool_calls_to_tool_message(message, tools: List[Tool], key_mappings: dict) -> AssistantMessage:
    from anthropic.types import TextBlock
    from anthropic.types.beta.tools import ToolUseBlock

    texts = [c.text for c in message.content if isinstance(c, TextBlock)]
    tool_uses = [c for c in message.content if isinstance(c, ToolUseBlock)]
    return AssistantMessage(
        name=", ".join([t.name for t in tool_uses]),
        content="\n\n".join(texts),
        tool_calls=[
            ToolCallMessage(
                id=t.id,
                name=t.name,
                payload=desanitize_json(t.input, key_mappings[t.name]),
                tool=next((item for item in tools if item.name == t.name), None)
            )
            for t in tool_uses
        ]
    )
