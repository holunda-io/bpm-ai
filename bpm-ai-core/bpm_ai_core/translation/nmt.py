from abc import ABC, abstractmethod

from bpm_ai_core.tracing.decorators import span


class NMTModel(ABC):
    """
    Neural Machine Translation Model
    """

    @abstractmethod
    async def _do_translate(self, text: str | list[str], target_language: str) -> str | list[str]:
        pass

    @span(name="nmt")
    async def translate(self, text: str | list[str], target_language: str) -> str | list[str]:
        return await self._do_translate(text, target_language)
