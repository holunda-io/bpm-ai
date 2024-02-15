import io
from abc import ABC, abstractmethod
from typing import Optional, Union

from bpm_ai_core.tracing.tracing import Tracing


class ASRModel(ABC):
    """
    Automatic Speech Recognition (ASR) model for transcribing audio.
    """

    @abstractmethod
    async def _transcribe(self, audio: Union[str, io.BytesIO], language: Optional[str] = None) -> str:
        pass

    async def transcribe(self, audio: Union[str, io.BytesIO], language: Optional[str] = None) -> str:
        Tracing.tracers().start_span("asr", inputs={"language": language})
        transcription = await self._transcribe(audio, language)
        Tracing.tracers().end_span(outputs={"transcription": transcription})
        return transcription
