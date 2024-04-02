from abc import ABC, abstractmethod
from typing import Tuple

from pydantic import BaseModel


class TokenClassificationResult(BaseModel):
    tags: list[Tuple[str, str]]


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
