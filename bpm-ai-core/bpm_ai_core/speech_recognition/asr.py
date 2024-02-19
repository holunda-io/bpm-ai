import io
from abc import ABC, abstractmethod
from typing import Optional, Union

from bpm_ai_core.tracing.tracing import Tracing
from bpm_ai_core.util.audio import load_audio


class ASRModel(ABC):
    """
    Automatic Speech Recognition (ASR) model for transcribing audio.
    """

    @abstractmethod
    async def _transcribe(self, audio: Union[str, io.BytesIO], language: Optional[str] = None) -> str:
        pass

    async def transcribe(self, audio_or_path: Union[io.BytesIO, str], language: Optional[str] = None) -> str:
        Tracing.tracers().start_span("asr", inputs={
            "language": language,
            "audio": audio_or_path if isinstance(audio_or_path, str) else "<io.BytesIO>"
        })
        if isinstance(audio_or_path, str):
            audio = load_audio(audio_or_path)
        else:
            audio = audio_or_path
        transcription = await self._transcribe(audio, language)
        Tracing.tracers().end_span(outputs={"transcription": transcription})
        return transcription
