import json
import logging
from typing import Dict, Any, Optional, List

from bpm_ai_core.llm.anthropic_chat import get_anthropic_client
from bpm_ai_core.llm.anthropic_chat._constants import DEFAULT_MODEL, DEFAULT_TEMPERATURE, \
    DEFAULT_MAX_RETRIES
from bpm_ai_core.llm.anthropic_chat.tools.tool import AnthropicTool
from bpm_ai_core.llm.anthropic_chat.tools.tool_user import ToolUser
from bpm_ai_core.llm.anthropic_chat.util import messages_to_anthropic_dicts
from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ChatMessage, ToolCallMessage, AssistantMessage, SystemMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.tracing.tracing import Tracing
from bpm_ai_core.util.json_schema import expand_simplified_json_schema

logger = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic, RateLimitError, InternalServerError, APIConnectionError
    from anthropic.types import Message

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
            result_dict = await self._run_output_schema_completion(messages, output_schema, current_try)
            return AssistantMessage(content=result_dict)
        elif tools:
            result_dict = await self._run_tool_completion(messages, tools)
            return self._tool_calls_to_tool_message(result_dict, tools)
        else:
            completion = await self._run_completion(messages, stop, current_try)
            return AssistantMessage(content=completion.content[0].text.strip())

    async def _run_completion(self, messages: List[ChatMessage], stop: list[str] = None, current_try: int = None) -> Message:
        Tracing.tracers().start_llm_trace(self, messages, current_try, None)
        completion = await self.client.messages.create(
            max_tokens=4096,
            model=self.model,
            temperature=self.temperature,
            system=messages.pop(0).content if (messages and messages[0].role == "system") else "",
            messages=messages_to_anthropic_dicts(messages),
            stop_sequences=stop
        )
        Tracing.tracers().end_llm_trace(completion.content[0].text)
        return completion

    async def _run_tool_completion(self, messages: list[ChatMessage], tools: list[Tool] = None) -> dict:
        tool_user = ToolUser(tools=[
            AnthropicTool(name=tool.name, description=tool.description, args_schema=tool.args_schema)
            for tool in tools
        ])
        return await tool_user.use_tools(
            messages=messages_to_anthropic_dicts(messages),
            execution_mode='manual',
            verbose=0
        )

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

    @staticmethod
    def _tool_calls_to_tool_message(result_dict: dict, tools: List[Tool]) -> AssistantMessage:
        return AssistantMessage(
            name=", ".join([t['tool_name'] for t in result_dict['tool_inputs']]),
            content=result_dict['content'],
            tool_calls=[
                ToolCallMessage(
                    id=t['tool_name'],
                    name=t['tool_name'],
                    payload=t['tool_arguments'],
                    tool=next((item for item in tools if item.name == t['tool_name']), None)
                )
                for t in result_dict['tool_inputs']
            ]
        )

    def supports_images(self) -> bool:
        return True

    def supports_video(self) -> bool:
        return False

    def supports_audio(self) -> bool:
        return False

    def name(self) -> str:
        return "anthropic"
