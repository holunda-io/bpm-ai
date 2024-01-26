import io
from typing import Optional, Dict, Any, Union

from bpm_ai_core.speech.stt.stt import STTModel
from bpm_ai_core.util.audio import load_audio

try:
    from openai import OpenAI
    has_openai = True
except ImportError:
    has_openai = False


class OpenAIWhisper(STTModel):
    """
    `OpenAI` Whisper STT API for transcribing audio.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.
    """

    def __init__(
        self,
        whisper_model: str = "whisper-1",
        client_kwargs: Optional[Dict[str, Any]] = None
    ):
        if not has_openai:
            raise ImportError('openai is not installed')
        self.whisper_model = whisper_model
        self.client = OpenAI(
            **(client_kwargs or {})
        )

    def transcribe(self, audio: Union[str, io.BytesIO], language: Optional[str] = None) -> str:
        if isinstance(audio, str):
            audio = load_audio(audio)
        transcript = self.client.audio.transcriptions.create(
            model=self.whisper_model,
            file=audio,
            **{"language": language} if language else {}
        )
        return transcript.text
