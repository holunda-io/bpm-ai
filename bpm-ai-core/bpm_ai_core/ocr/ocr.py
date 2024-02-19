from abc import ABC, abstractmethod

from PIL.Image import Image
from pydantic import BaseModel

from bpm_ai_core.tracing.tracing import Tracing
from bpm_ai_core.util.image import load_images


class OCRResult(BaseModel):
    texts: list[str]


class OCR(ABC):
    """
    Optical Character Recognition (OCR) Model
    """

    @abstractmethod
    async def images_to_text_with_metadata(
            self,
            images: list[Image],
            language: str = None
    ) -> OCRResult:
        pass

    async def images_to_text(
            self,
            image_or_path: Image | list[Image] | str,
            language: str = None
    ) -> str:
        Tracing.tracers().start_span("ocr", inputs={
            "image": image_or_path,
            "language": language
        })
        if isinstance(image_or_path, str):
            images = load_images(image_or_path)
        elif isinstance(image_or_path, Image):
            images = [image_or_path]
        else:  # list[Image]
            images = image_or_path
        result = await self.images_to_text_with_metadata(
            images=images,
            language=language
        )
        Tracing.tracers().end_span(outputs={"result": result.model_dump()})
        return "\n".join(result.texts)
