from abc import ABC, abstractmethod
from typing import Tuple

from pydantic import BaseModel

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.tracing.decorators import span
from bpm_ai_core.util.caching import cached, blob_cache_key, cachable


class ClassificationResult(BaseModel):
    max_label: str
    max_score: float
    labels_scores: list[Tuple[str, float]]


class ImageClassifier(ABC):
    """
    (Zero Shot) Image Classification Model
    """

    @abstractmethod
    async def _do_classify(
            self,
            blob: Blob,
            classes: list[str] = None,
            hypothesis_template: str | None = None,
    ) -> ClassificationResult:
        pass

    @cached(exclude=["blob_or_path"], key_func=blob_cache_key)
    @span(name="image-classifier")
    async def classify(
            self,
            blob_or_path: Blob | str,
            classes: list[str] = None,
            hypothesis_template: str | None = None,
            confidence_threshold: float = None,
    ) -> ClassificationResult:
        if isinstance(blob_or_path, str):
            blob = Blob.from_path_or_url(blob_or_path)
        else:  # Blob
            blob = blob_or_path
        result = await self._do_classify(
            blob=blob,
            classes=classes,
            hypothesis_template=hypothesis_template
        )
        # Only return if the score is above the threshold (if given)
        return result \
            if result.max_score > (confidence_threshold or -1) \
            else None
