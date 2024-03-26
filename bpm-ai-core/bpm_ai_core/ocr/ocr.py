from abc import ABC, abstractmethod
from typing import Tuple

from pydantic import BaseModel, Field

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.tracing.decorators import span


class OCRPage(BaseModel):
    text: str
    words: list[str] = Field(..., exclude=True)
    bboxes: list[Tuple[float, float, float, float]] = Field(..., exclude=True)
    """Format: (x, y, x + w, y + h), normalized to 1"""


class OCRResult(BaseModel):
    pages: list[OCRPage]

    @property
    def full_text(self) -> str:
        return "\n".join([p.text for p in self.pages])


class OCR(ABC):
    """
    Optical Character Recognition (OCR) Model
    """

    @abstractmethod
    async def _do_process(
            self,
            blob: Blob,
            language: str = None
    ) -> OCRResult:
        pass

    @span(name="ocr")
    async def process(
            self,
            blob_or_path: Blob | str,
            language: str = None
    ) -> OCRResult:
        if isinstance(blob_or_path, str):
            blob = Blob.from_path_or_url(blob_or_path)
        else:  # Blob
            blob = blob_or_path
        return await self._do_process(
            blob=blob,
            language=language
        )
