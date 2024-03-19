import io

from bpm_ai_core.speech_recognition.asr import ASRModel

try:
    from faster_whisper import WhisperModel
    has_faster_whisper = True
except ImportError:
    has_faster_whisper = False


class FasterWhisperASR(ASRModel):
    """
    Local `OpenAI` Whisper Automatic Speech Recognition (ASR) model for transcribing audio.

    To use, you should have the ``faster_whisper`` python package installed.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        if not has_faster_whisper:
            raise ImportError('faster_whisper is not installed')
        self.model_size = model_size
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    async def _transcribe(self, audio: io.BytesIO, language: str = None) -> str:
        segments, info = self.model.transcribe(audio, language=language)
        return "".join([s.text for s in list(segments)])
