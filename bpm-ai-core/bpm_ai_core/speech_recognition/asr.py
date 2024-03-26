import io
from abc import ABC, abstractmethod
from typing import Optional, Union

from bpm_ai_core.tracing.decorators import span
from bpm_ai_core.util.audio import load_audio


class ASRModel(ABC):
    """
    Automatic Speech Recognition (ASR) model for transcribing audio.
    """

    @abstractmethod
    async def _do_transcribe(self, audio: io.BytesIO, language: Optional[str] = None) -> str:
        pass

    @span(name="asr")
    async def transcribe(self, audio_or_path: Union[io.BytesIO, str], language: Optional[str] = None) -> str:
        if isinstance(audio_or_path, str):
            audio = load_audio(audio_or_path)
        else:
            audio = audio_or_path
        return await self._do_transcribe(audio, language)
