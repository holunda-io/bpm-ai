from abc import ABC, abstractmethod
from typing import Tuple

from pydantic import BaseModel

from bpm_ai_core.tracing.decorators import span


class ClassificationResult(BaseModel):
    max_label: str
    max_score: float
    labels_scores: list[Tuple[str, float]]


class ZeroShotClassifier(ABC):
    """
    Zero Shot Classification Model
    """

    @abstractmethod
    async def _do_classify(
            self,
            text: str,
            classes: list[str],
            hypothesis_template: str | None = None
    ) -> ClassificationResult:
        pass

    @span(name="classifier")
    async def classify(
            self,
            text: str,
            classes: list[str],
            confidence_threshold: float | None = None,
            hypothesis_template: str | None = None
    ) -> ClassificationResult:
        result = await self._do_classify(
            text=text,
            classes=classes,
            hypothesis_template=hypothesis_template
        )
        # Only return if the score is above the threshold (if given)
        return result \
            if not confidence_threshold or result.max_score > confidence_threshold \
            else None
