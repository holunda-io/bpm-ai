from typing import List, Any

from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ChatMessage, ToolCallMessage, AssistantMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.tracing.tracing import Tracing


def messages_to_str(messages: List[ChatMessage]) -> str:
    return "\n\n".join([m.content for m in messages])


def tool_response(name, payload):
    return AssistantMessage(tool_calls=[ToolCallMessage(id="fake", name=name, payload=payload)])


class FakeLLM(LLM):
    """Fake LLM for testing purposes."""

    def __init__(
        self,
        responses: list[AssistantMessage] | None = None,
        tools: list[list[Tool]] | None = None,
        supports_images: bool = False,
        supports_video: bool = False,
        supports_audio: bool = False,
        name: str = "test-llm"
    ):
        super().__init__("test-model")
        self._name = name

        self.responses = responses or []
        self.tools = tools or []
        self.response_idx: int = 0
        self.requests = []

        self._supports_images = supports_images
        self._supports_video = supports_video
        self._supports_audio = supports_audio

    async def _generate_message(
        self, messages: list[ChatMessage],
        output_schema: dict[str, Any] | None = None,
        tools: list[Tool] | None = None,
        stop: list[str] = None,
        current_try: int = None
    ) -> AssistantMessage:
        self.requests += [messages]
        self.tools += [tools]

        Tracing.tracers().start_llm_trace(self, messages, current_try, tools)

        if self.responses:
            response = self.responses[self.response_idx]
            if response.has_tool_calls():
                for c in response.tool_calls:
                    c.tool = next((item for item in tools if item.name == c.name), None)
            self.response_idx += 1
        else:
            response = None

        Tracing.tracers().end_llm_trace(response)

        return response

    def assert_last_request_contains(self, text: str):
        assert text in messages_to_str(self.requests[-1])

    def assert_last_request_not_contains(self, text: str):
        assert text not in messages_to_str(self.requests[-1])

    def assert_no_request(self):
        assert len(self.requests) == 0

    def assert_last_request_defined_tool(self, tool_name: str, is_fixed_tool_choice: bool = False):
        assert tool_name in [f.name for f in self.tools[-1]]
        if is_fixed_tool_choice:
            assert len(self.tools[-1]) == 1 and self.tools[-1][0].name == tool_name

    def supports_images(self) -> bool:
        return self._supports_images

    def supports_video(self) -> bool:
        return self._supports_video

    def supports_audio(self) -> bool:
        return self._supports_audio

    def name(self) -> str:
        return self._name
