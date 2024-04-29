import io
from abc import ABC, abstractmethod
from typing import Optional, Union

from pydantic import BaseModel

from bpm_ai_core.tracing.decorators import span
from bpm_ai_core.util.audio import load_audio
from bpm_ai_core.util.caching import cached


class ASRResult(BaseModel):
    text: str


class ASRModel(ABC):
    """
    Automatic Speech Recognition (ASR) model for transcribing audio.
    """

    @abstractmethod
    async def _do_transcribe(self, audio: io.BytesIO, language: Optional[str] = None) -> ASRResult:
        pass

    @staticmethod
    async def _cache_key(audio_or_path: io.BytesIO | str, *args, **kwargs) -> str:
        if isinstance(audio_or_path, str):
            return f"path={audio_or_path}"
        else:  # BytesIO
            return f"audio_bytes={audio_or_path.getvalue()}"

    @cached(exclude=["audio_or_path"], key_func=_cache_key)
    @span(name="asr")
    async def transcribe(self, audio_or_path: io.BytesIO | str, language: Optional[str] = None) -> ASRResult:
        if isinstance(audio_or_path, str):
            audio = load_audio(audio_or_path)
        else:
            audio = audio_or_path
        return await self._do_transcribe(audio, language)
