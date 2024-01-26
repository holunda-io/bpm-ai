from typing import Dict, Any, Optional, List

from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ChatMessage
from bpm_ai_core.llm.common.tool import Tool

try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    has_anthropic = True
except ImportError:
    has_anthropic = False


class AnthropicChat(LLM):

    def __init__(
        self,
        model: str = "claude-2.1",
        temperature: float = 0.0,
        max_retries: int = 8,
        client_kwargs: Optional[Dict[str, Any]] = None
    ):
        if not has_anthropic:
            raise ImportError('anthropic is not installed')
        super().__init__(
            model=model,
            temperature=temperature,
            max_retries=max_retries
        )
        self.client = Anthropic(
            max_retries=0,  # we use own retry logic
            **(client_kwargs or {})
        )

    def _predict(
        self,
        messages: List[ChatMessage],
        output_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None
    ) -> ChatMessage:

        completion = self._run_completion(messages, [])

        return ChatMessage(
            role="assistant",
            content=completion.completion.strip()
        )

    def _run_completion(self, messages: List[ChatMessage], functions: List[dict]):
        prompt = AnthropicChat.messages_to_claude_str(messages)
        return self.client.completions.create(
            max_tokens_to_sample=4096,
            model=self.model,
            temperature=self.temperature,
            prompt=prompt
        )

    @staticmethod
    def messages_to_claude_str(messages: List[ChatMessage]) -> str:
        return "".join([AnthropicChat.message_to_claude_str(m) for m in messages] + [AI_PROMPT])

    @staticmethod
    def message_to_claude_str(m) -> str:
        if m.role == "assistant":
            header = AI_PROMPT + " "
        elif m.role == "user":
            header = HUMAN_PROMPT + " "
        else:
            header = ""
        return header + m.content

    def name(self) -> str:
        return "anthropic"
