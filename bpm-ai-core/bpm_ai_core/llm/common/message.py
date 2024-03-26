import asyncio
import inspect
import json
from typing import Optional, Literal, Any, Union, List

from pydantic import BaseModel, Field, ConfigDict

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.tracing.tracing import Tracing


class ChatMessage(BaseModel):
    content: Optional[Union[str, dict, List[Union[str, Blob]]]] = None
    """
    The contents of the message. 
    Either a string for normal completions, 
    or a list of strings and blobs for multimodal completions, 
    or a dict for prediction with output schema.
    """

    role: Literal["system", "user", "assistant", "tool"]
    """
    The role of the messages author.
    One of `system`, `user`, `assistant`, or `tool`.
    """

    name: Optional[str] = None
    """
    The name of the author of this message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SystemMessage(ChatMessage):

    role: str = Field("system")


class UserMessage(ChatMessage):

    role: str = Field("user")


class ToolCallMessage(BaseModel):

    id: str

    type: str = Field("function")

    name: str
    """
    The name of the tool that was used.
    """

    payload: Any

    tool: Optional[Tool] = None

    def payload_dict(self) -> dict:
        if isinstance(self.payload, dict):
            return self.payload
        elif isinstance(self.payload, str):
            try:
                return json.loads(self.payload)
            except ValueError as e:
                raise Exception(f"Payload could not be converted to a dict: {e}")
        else:
            raise Exception("Payload has unexpected type.")

    async def run_tool_function(self) -> Any:
        _callable = self.tool.callable
        inputs = self.payload_dict()
        Tracing.tracers().start_tool_trace(self.tool, inputs)
        if inspect.iscoroutinefunction(_callable):
            result = await _callable(**inputs)
        else:
            result = _callable(**inputs)
        Tracing.tracers().end_tool_trace(result)
        return result


class ToolResultMessage(ChatMessage):

    role: str = Field("tool")

    id: str
    """
    The id of the tool.
    """


class AssistantMessage(ChatMessage):

    role: str = Field("assistant")

    tool_calls: List[ToolCallMessage] | None = None

    def has_tool_calls(self) -> bool:
        return self.tool_calls and len(self.tool_calls) > 0

    async def run_all_tool_functions(self) -> List[Any]:
        return [await t.run_tool_function() for t in self.tool_calls]

    async def run_all_tool_functions_parallel(self) -> List[Any]:
        return await asyncio.gather(*[t.run_tool_function() for t in self.tool_calls])
