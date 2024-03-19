import logging
from typing import List

from PIL.Image import Image

from bpm_ai_core.llm.common.message import ChatMessage, ToolResultMessage, AssistantMessage
from bpm_ai_core.util.image import base64_encode_image

logger = logging.getLogger(__name__)


def messages_to_anthropic_dicts(messages: List[ChatMessage]):
    return [message_to_anthropic_dict(m) for m in messages]


def message_to_anthropic_dict(message: ChatMessage) -> dict:
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
            elif isinstance(e, Image):
                content.append(image_to_anthropic_image_dict(e))
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
        **({"name": message.name} if message.name else {})
    }


def tool_calls_message_to_anthropic_dict(message: AssistantMessage) -> dict:
    return {
        "role": "tool_inputs",
        "content": message.content,
        "tool_inputs": [
            {
                "tool_name": call.name,
                "tool_arguments": call.payload_dict()
            }
            for call in message.tool_calls
        ]
    }


def tool_result_message_to_anthropic_dict(message: ToolResultMessage) -> dict:
    return {
        "role": "tool_outputs",
        "tool_outputs": [
            {
                "tool_name": message.id,
                "tool_result": message.content
            }
        ],
        "tool_error": None
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