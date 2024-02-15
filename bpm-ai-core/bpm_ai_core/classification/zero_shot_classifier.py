from abc import ABC, abstractmethod
from typing import Tuple

from pydantic import BaseModel

from bpm_ai_core.tracing.tracing import Tracing


class ClassificationResult(BaseModel):
    max_label: str
    max_score: float
    labels_scores: list[Tuple[str, float]]


class ZeroShotClassifier(ABC):
    """
    Zero Shot Classification Model
    """

    @abstractmethod
    def classify_with_metadata(
            self,
            text: str,
            classes: list[str],
            hypothesis_template: str | None = None
    ) -> ClassificationResult:
        pass

    def classify(
            self,
            text: str,
            classes: list[str],
            confidence_threshold: float | None = None,
            hypothesis_template: str | None = None
    ) -> str:
        Tracing.tracers().start_span("classification", inputs={
            "text": text,
            "classes": classes,
            "confidence_threshold": confidence_threshold,
            "hypothesis_template": hypothesis_template
        })
        result = self.classify_with_metadata(
            text=text,
            classes=classes,
            hypothesis_template=hypothesis_template
        )
        Tracing.tracers().end_span(outputs={"result": result.model_dump()})
        # Only return the label if the score is above the threshold (if given)
        return result.max_label \
            if not confidence_threshold or result.max_score > confidence_threshold \
            else None
