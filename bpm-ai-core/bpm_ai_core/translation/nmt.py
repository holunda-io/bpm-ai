from abc import ABC, abstractmethod

from bpm_ai_core.tracing.tracing import Tracing


class NMTModel(ABC):
    """
    Neural Machine Translation Model
    """

    @abstractmethod
    def _translate(self, text: str | list[str], target_language: str) -> str | list[str]:
        pass

    def translate(self, text: str | list[str], target_language: str) -> str | list[str]:
        Tracing.tracers().start_span("nmt", inputs={
            "text": text,
            "target_language": target_language
        })
        translation = self._translate(text, target_language)
        Tracing.tracers().end_span(outputs={"translation": translation})
        return translation
