from abc import ABC, abstractmethod

from pydantic import BaseModel

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.tracing.decorators import span
from bpm_ai_core.util.caching import cached, calculate_cache_key


class QAResult(BaseModel):
    answer: str | None
    score: float
    start_index: int | None
    end_index: int | None


class QuestionAnswering(ABC):
    """
    (Extractive) Question Answering Model
    """

    @abstractmethod
    async def _do_answer(
        self,
        context_str_or_blob: str | Blob,
        question: str
    ) -> QAResult:
        pass

    async def _cache_key(self, context_str_or_blob: str | Blob, *args, **kwargs) -> str:
        if isinstance(context_str_or_blob, str):
            return f"context={context_str_or_blob}"
        else:  # Blob
            return f"blob_bytes={await context_str_or_blob.as_bytes()}"

    @cached(exclude=["context_str_or_blob"])
    @span(name="qa")
    async def answer(
        self,
        context_str_or_blob: str | Blob,
        question: str,
        confidence_threshold: float | None = 0.1
    ) -> QAResult | None:
        result = await self._do_answer(
            context_str_or_blob=context_str_or_blob,
            question=question
        )
        # Only return the answer if the score is above the threshold (if given)
        return result \
            if not confidence_threshold or result.score > confidence_threshold \
            else None
