from abc import abstractmethod, ABC
from typing import Any, Type

from tenacity import stop_after_attempt, wait_exponential, retry_if_exception_type, Retrying

from bpm_ai_core.llm.common.message import ChatMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.tracing.tracing import Tracing


class LLM(ABC):
    """
    Abstract class for large language models.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_retries: int = 8,
        retryable_exceptions: list[Type[BaseException]] = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retryable_exceptions = retryable_exceptions or [Exception]

    async def predict(
        self,
        prompt: Prompt,
        output_schema: dict[str, Any] | None = None,
        tools: list[Tool] | None = None
    ) -> ChatMessage:
        if output_schema and tools:
            raise ValueError("Must not pass both an output_schema and tools")

        messages = prompt.format(llm_name=self.name())

        for attempt in Retrying(
            wait=wait_exponential(multiplier=1.5, min=2, max=60),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception_type(tuple(self.retryable_exceptions))
        ):
            with attempt:
                Tracing.tracers().start_llm_trace(self, messages, attempt.retry_state.attempt_number, tools)
                completion = await self._predict(messages, output_schema, tools)
                Tracing.tracers().end_llm_trace(completion)

        return completion

    @abstractmethod
    async def _predict(
        self,
        messages: list[ChatMessage],
        output_schema: dict[str, Any] | None = None,
        tools: list[Tool] | None = None
    ) -> ChatMessage:
        pass

    @abstractmethod
    def supports_images(self) -> bool:
        pass

    @abstractmethod
    def supports_audio(self) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
