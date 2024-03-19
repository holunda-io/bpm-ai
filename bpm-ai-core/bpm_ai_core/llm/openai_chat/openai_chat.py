import json
import logging
import os
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin

from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ChatMessage, ToolCallMessage, AssistantMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.llm.openai_chat import get_openai_client, get_azure_openai_client
from bpm_ai_core.llm.openai_chat._constants import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_SEED, \
    DEFAULT_MAX_RETRIES, AZURE_API_KEY_ENV_VAR, OPENAI_COMPATIBLE_API_KEY_ENV_VAR
from bpm_ai_core.llm.openai_chat.util import messages_to_openai_dicts, json_schema_to_openai_function
from bpm_ai_core.tracing.tracing import Tracing

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, APIConnectionError, InternalServerError, RateLimitError, OpenAIError
    from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage, ChatCompletion
    from openai.lib.azure import AsyncAzureOpenAI

    has_openai = True
except ImportError:
    has_openai = False


class ChatOpenAI(LLM):
    """
    `OpenAI` Chat large language models API.
    Also used for Azure OpenAI and other compatible servers.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: Optional[int] = DEFAULT_SEED,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: AsyncOpenAI = None
    ):
        if not has_openai:
            raise ImportError('openai is not installed')
        if not client:
            client = get_openai_client()
        super().__init__(
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            retryable_exceptions=[
                RateLimitError, InternalServerError, APIConnectionError
            ]
        )
        self.client = client
        self.seed = seed

    @classmethod
    def for_openai(
        cls,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: Optional[int] = DEFAULT_SEED,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        return cls(
            model=model,
            temperature=temperature,
            seed=seed,
            max_retries=max_retries
        )

    @classmethod
    def for_openai_compatible(
        cls,
        endpoint: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: Optional[int] = DEFAULT_SEED,
        max_retries: int = DEFAULT_MAX_RETRIES,
        api_key_env_var: str = OPENAI_COMPATIBLE_API_KEY_ENV_VAR
    ):
        return cls(
            model=model,
            temperature=temperature,
            seed=seed,
            max_retries=max_retries,
            client=get_openai_client(
                endpoint=urljoin(endpoint, "/v1"),
                api_key=os.environ.get(api_key_env_var, "dummy")
            )
        )

    @classmethod
    def for_azure(
        cls,
        endpoint: str,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: Optional[int] = DEFAULT_SEED,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        pattern = r"(https://[^/]+)/openai/deployments/([^/]+)/[^/]+/[^/]+\?api-version=([^&]+)"
        match = re.search(pattern, endpoint)
        if match:
            azure_endpoint = match.group(1)
            deployment = match.group(2)
            api_version = match.group(3)
        else:
            raise Exception("Full endpoint with deployment and api-version required")
        api_key = os.environ.get(AZURE_API_KEY_ENV_VAR)
        if not api_key:
            raise Exception(f"API key not found ({AZURE_API_KEY_ENV_VAR} env variable)")
        return cls(
            model=deployment,
            temperature=temperature,
            seed=seed,
            max_retries=max_retries,
            client=get_azure_openai_client(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key
            )
        )

    async def _generate_message(
        self,
        messages: List[ChatMessage],
        output_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        stop: list[str] = None,
        current_try: int = None
    ) -> AssistantMessage:
        tools = [self._output_schema_to_tool(output_schema)] if output_schema else tools
        openai_tools = [json_schema_to_openai_function(f.name, f.description, f.args_schema) for f in tools] if tools else []
        completion = await self._run_completion(messages, openai_tools, stop, current_try)
        message = completion.choices[0].message
        if message.tool_calls:
            if output_schema:
                return AssistantMessage(content=self._parse_tool_call_json(message))
            else:
                return self._openai_tool_calls_to_tool_message(message, tools)
        else:
            return AssistantMessage(content=message.content)

    async def _run_completion(
        self,
        messages: List[ChatMessage],
        tools: List[dict],
        stop: list[str] = None,
        current_try: int = None
    ) -> ChatCompletion:
        args = {
            "model": self.model,
            "temperature": self.temperature,
            **({"seed": self.seed} if self.seed else {}),
            "messages": messages_to_openai_dicts(messages),
            "stop": stop or [],
            **({
                   "tool_choice": {
                       "type": "function",
                       "function": {"name": tools[0]["function"]["name"]}
                   } if (len(tools) == 1) else ("auto" if tools else "none"),
                   "tools": tools
               } if tools else {})
        }
        Tracing.tracers().start_llm_trace(self, messages, current_try, tools)
        completion = await self.client.chat.completions.create(**args)
        Tracing.tracers().end_llm_trace(completion.choices[0].message)
        return completion

    @staticmethod
    def _output_schema_to_tool(output_schema: dict):
        output_schema = output_schema.copy()
        return Tool.create(
            name=output_schema.pop("name") or "store_result",
            description=output_schema.pop("description") or "Stores your result",
            args_schema=output_schema
        )

    @staticmethod
    def _openai_tool_calls_to_tool_message(message: OpenAIChatCompletionMessage, tools: List[Tool]) -> AssistantMessage:
        return AssistantMessage(
            name=", ".join([t.function.name for t in message.tool_calls]),
            content=message.content,
            tool_calls=[
                ToolCallMessage(
                    id=t.id,
                    name=t.function.name,
                    payload=t.function.arguments,
                    tool=next((item for item in tools if item.name == t.function.name), None)
                )
                for t in message.tool_calls
            ]
        )

    @staticmethod
    def _parse_tool_call_json(message: OpenAIChatCompletionMessage):
        try:
            json_object = json.loads(message.tool_calls[0].function.arguments)
        except ValueError as e:
            json_object = None
        return json_object

    def supports_images(self) -> bool:
        return "vision" in self.model

    def supports_video(self) -> bool:
        return False

    def supports_audio(self) -> bool:
        return False

    def name(self) -> str:
        return "openai"
