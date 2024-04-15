import json
import logging
from typing import Dict, Any, Optional, List

from typing_extensions import deprecated

from bpm_ai_core.llm.anthropic_chat import get_anthropic_client
from bpm_ai_core.llm.anthropic_chat._constants import DEFAULT_MODEL, DEFAULT_TEMPERATURE, \
    DEFAULT_MAX_RETRIES
from bpm_ai_core.llm.anthropic_chat.util import messages_to_anthropic_dicts, json_schema_to_anthropic_tool, \
    tool_calls_to_tool_message
from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ChatMessage, ToolCallMessage, AssistantMessage, SystemMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.tracing.tracing import Tracing
from bpm_ai_core.util.json_schema import expand_simplified_json_schema

logger = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic, RateLimitError, InternalServerError, APIConnectionError
    from anthropic.types import Message, TextBlock
    from anthropic.types.beta.tools import ToolUseBlock, ToolsBetaMessage

    has_anthropic = True
except ImportError:
    has_anthropic = False


class ChatAnthropic(LLM):
    """
    `Anthropic` Chat large language models API.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: AsyncAnthropic = None
    ):
        if not has_anthropic:
            raise ImportError('anthropic is not installed')
        if not client:
            client = get_anthropic_client()
        super().__init__(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            retryable_exceptions=[
                RateLimitError, InternalServerError, APIConnectionError
            ]
        )
        self.client = client

    @classmethod
    def for_anthropic(
        cls,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        return cls(
            model=model,
            temperature=temperature,
            max_retries=max_retries
        )

    async def _generate_message(
        self,
        messages: List[ChatMessage],
        output_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        stop: list[str] = None,
        current_try: int = None
    ) -> AssistantMessage:
        if output_schema:
            tools = [self._output_schema_to_tool(output_schema)]
            message = await self._run_tool_completion(messages, tools, current_try)
            if message.has_tool_calls():
                return AssistantMessage(content=message.tool_calls[0].payload)
            else:
                return AssistantMessage(content=None)
        elif tools:
            return await self._run_tool_completion(messages, tools, current_try)
        else:
            completion = await self._run_completion(messages, stop, current_try)
            return AssistantMessage(content=completion.content[0].text.strip())

    async def _run_completion(self, messages: List[ChatMessage], stop: list[str] = None, current_try: int = None) -> Message:
        messages = await messages_to_anthropic_dicts(messages)
        Tracing.tracers().start_llm_trace(self, messages, current_try, None)
        completion = await self.client.messages.create(
            max_tokens=4096,
            model=self.model,
            temperature=self.temperature,
            system=messages.pop(0)["content"] if (messages and messages[0]["role"] == "system") else "",
            messages=messages,
            stop_sequences=stop
        )
        Tracing.tracers().end_llm_trace(completion)
        return completion

    async def _run_tool_completion(self, messages: list[ChatMessage], tools: list[Tool] = None, current_try: int = None) -> AssistantMessage:
        sanitized_key_mappings = {t.name: {} for t in tools}
        anthropic_tools = [json_schema_to_anthropic_tool(t.name, t.description, t.args_schema, sanitized_key_mappings[t.name]) for t in tools] if tools else []
        Tracing.tracers().start_llm_trace(self, messages, current_try, anthropic_tools)
        completion = await self.client.beta.tools.messages.create(
            max_tokens=4096,
            model=self.model,
            temperature=self.temperature,
            system=messages.pop(0).content if (messages and messages[0].role == "system") else "",
            messages=await messages_to_anthropic_dicts(messages),
            tools=anthropic_tools
        )
        Tracing.tracers().end_llm_trace(completion)
        return tool_calls_to_tool_message(completion, tools, sanitized_key_mappings)

    @staticmethod
    def _output_schema_to_tool(output_schema: dict):
        output_schema = output_schema.copy()
        return Tool.create(
            name=output_schema.pop("name", None) or "record_result",
            description=output_schema.pop("description", None) or "Record your result into well-structured JSON.",
            args_schema=output_schema
        )

    @deprecated("Using a tool for now but still need to evaluate which is better")
    async def _run_output_schema_completion(self, messages: list[ChatMessage], output_schema: dict[str, Any], current_try: int = None) -> dict:
        output_schema = expand_simplified_json_schema(output_schema)
        output_prompt = Prompt.from_file(
            "output_schema",
            output_schema=json.dumps(output_schema, indent=2)
        ).format()[0].content
        if messages[0].role == "system":
            messages[0].content += f"\n\n{output_prompt}"
        else:
            messages.insert(0, SystemMessage(content=output_prompt))
        if messages[-1].role == "assistant":
            logger.warning("Ignoring trailing assistant message.")
            messages.pop()
        messages.append(AssistantMessage(content="<result>"))
        completion = await self._run_completion(messages, stop=["</result>"], current_try=current_try)
        try:
            json_object = json.loads(completion.content[0].text.strip())
        except ValueError:
            json_object = None
        return json_object

    def supports_images(self) -> bool:
        return True

    def supports_video(self) -> bool:
        return False

    def supports_audio(self) -> bool:
        return False

    def name(self) -> str:
        return "anthropic"
