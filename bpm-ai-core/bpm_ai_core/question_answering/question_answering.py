from abc import ABC, abstractmethod

from PIL.Image import Image
from pydantic import BaseModel

from bpm_ai_core.tracing.tracing import Tracing


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
    def answer_with_metadata(
            self,
            context: str | Image,
            question: str
    ) -> QAResult:
        pass

    def answer(
            self,
            context: str | Image,
            question: str,
            confidence_threshold: float | None = 0.1
    ) -> str:
        Tracing.tracers().start_span("qa", inputs={
            "context": context,
            "question": question,
            "confidence_threshold": confidence_threshold
        })
        result = self.answer_with_metadata(
            context=context,
            question=question
        )
        Tracing.tracers().end_span(outputs={"result": result.model_dump()})
        # Only return the answer if the score is above the threshold (if given)
        return result.answer \
            if not confidence_threshold or result.score > confidence_threshold \
            else None
