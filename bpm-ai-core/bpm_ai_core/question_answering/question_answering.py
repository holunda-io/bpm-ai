from abc import ABC, abstractmethod

from pydantic import BaseModel

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.tracing.decorators import span


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
