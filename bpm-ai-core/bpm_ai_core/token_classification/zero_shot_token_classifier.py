from abc import ABC, abstractmethod

from pydantic import BaseModel


class TokenSpan(BaseModel):
    label: str
    score: float
    word: str
    start: int
    end: int


class TokenClassificationResult(BaseModel):
    spans: list[TokenSpan]


class ZeroShotTokenClassifier(ABC):
    """
    Zero Shot Token Classification Model
    """

    @abstractmethod
    async def _do_classify(
            self,
            text: str,
            classes: list[str],
            confidence_threshold: float | None = None
    ) -> TokenClassificationResult:
        pass

    async def classify(
            self,
            text: str,
            classes: list[str],
            confidence_threshold: float | None = None
    ) -> TokenClassificationResult:
        return await self._do_classify(
            text=text,
            classes=classes,
            confidence_threshold=confidence_threshold
        )
