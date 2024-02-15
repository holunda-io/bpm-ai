import io
from typing import Optional, Dict, Any, Union
import logging

from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.util.audio import load_audio

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    import httpx

    has_openai = True
    try:
        client = AsyncOpenAI(
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )
    except OpenAIError as e:
        logger.error(e)
except ImportError:
    has_openai = False


class OpenAIWhisperASR(ASRModel):
    """
    `OpenAI` Whisper Automatic Speech Recognition (ASR) API for transcribing audio.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.
    """

    def __init__(
        self,
        whisper_model: str = "whisper-1"
    ):
        if not has_openai:
            raise ImportError('openai is not installed')
        self.whisper_model = whisper_model

    async def _transcribe(self, audio: Union[str, io.BytesIO], language: Optional[str] = None) -> str:
        if isinstance(audio, str):
            audio = load_audio(audio)
        transcript = await client.audio.transcriptions.create(
            model=self.whisper_model,
            file=audio,
            **{"language": language} if language else {}
        )
        return transcript.text
