from abc import ABC, abstractmethod
from typing import Tuple

from pydantic import BaseModel

from bpm_ai_core.tracing.decorators import span
from bpm_ai_core.util.caching import cached


class ClassificationResult(BaseModel):
    max_label: str
    max_score: float
    labels_scores: list[Tuple[str, float]]


class TextClassifier(ABC):
    """
    (Zero Shot) Text Classification Model
    """

    @abstractmethod
    async def _do_classify(
            self,
            text: str,
            classes: list[str] = None,
            hypothesis_template: str | None = None,
            multi_label: bool = False
    ) -> ClassificationResult:
        pass

    @cached()
    @span(name="classifier")
    async def classify(
            self,
            text: str,
            classes: list[str] = None,
            confidence_threshold: float | None = None,
            hypothesis_template: str | None = None,
            multi_label: bool = False
    ) -> ClassificationResult:
        result = await self._do_classify(
            text=text,
            classes=classes,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label
        )
        if multi_label:
            result.labels_scores = [(l, s) for l, s in result.labels_scores if s > (confidence_threshold or -1)]

        # Only return if the score is above the threshold (if given)
        return result \
            if result.max_score > (confidence_threshold or -1) \
            else None
