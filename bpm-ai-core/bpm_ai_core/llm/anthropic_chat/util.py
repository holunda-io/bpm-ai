import logging
from typing import List, Any

from PIL.Image import Image

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.llm.common.message import ChatMessage, ToolResultMessage, AssistantMessage
from bpm_ai_core.util.image import base64_encode_image, blob_as_images

logger = logging.getLogger(__name__)


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
                images = await blob_as_images(e, accept_formats=["jpg", "png"])
                for image in images:
                    content.append(image_to_anthropic_image_dict(image))
            else:
                raise ValueError(
                    "Elements in ChatMessage.content must be str or image Blob"
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


def json_schema_to_anthropic_tool(name: str, desc: str, schema: dict[str, Any]) -> dict:
    return {
        "name": name,
        "description": desc,
        "input_schema": schema
    }