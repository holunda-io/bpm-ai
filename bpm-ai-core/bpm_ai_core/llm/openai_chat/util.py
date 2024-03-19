import logging
from typing import List, Dict, Any

from PIL.Image import Image

from bpm_ai_core.llm.common.message import ChatMessage, ToolResultMessage, AssistantMessage
from bpm_ai_core.util.image import base64_encode_image

logger = logging.getLogger(__name__)


def get_openai_tool_call_dict(message: AssistantMessage):
    return {
        "tool_calls": [
            {
                "type": "function",
                "id": t.id,
                "function": {
                    "name": t.name,
                    "arguments": t.payload
                }
            }
            for t in message.tool_calls
        ]
    }


def messages_to_openai_dicts(messages: List[ChatMessage]):
    return [message_to_openai_dict(m) for m in messages]


def message_to_openai_dict(message: ChatMessage) -> dict:
    if isinstance(message, AssistantMessage) and message.has_tool_calls():
        extra_dict = {
            **get_openai_tool_call_dict(message)
        }
    elif isinstance(message, ToolResultMessage):
        extra_dict = {
            "tool_call_id": message.id
        }
    else:
        extra_dict = {}

    if isinstance(message.content, str):
        content = message.content
    elif isinstance(message.content, list):
        content = []
        for e in message.content:
            if isinstance(e, str):
                content.append(str_to_openai_text_dict(e))
            elif isinstance(e, Image):
                content.append(image_to_openai_image_dict(e))
            else:
                raise ValueError(
                    "Elements in ChatMessage.content must be of type str or PIL.Image."
                )
    else:
        content = None
        logger.warning(
            "ChatMessage.content must be of type str or List[Union[str, PIL.Image]] if used for chat completions."
        )
    return {
        "role": message.role,
        **({"content": content} if content else {}),
        **extra_dict,
        **({"name": message.name} if message.name else {})
    }


def image_to_openai_image_dict(image: Image) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/{image.format.lower()};base64,{base64_encode_image(image)}"
        }
    }


def str_to_openai_text_dict(text: str) -> dict:
    return {
        "type": "text",
        "text": text
    }


def json_schema_to_openai_function(name: str, desc: str, schema: Dict[str, Any]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": schema
        }
    }
