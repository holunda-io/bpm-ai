from abc import ABC, abstractmethod
from typing import Tuple

from pydantic import BaseModel


class POSResult(BaseModel):
    tags: list[Tuple[str, str]]


class POSTagger(ABC):
    """
    Part-of-Speech Tagging Model
    """

    @abstractmethod
    async def _do_tag(self, text: str) -> POSResult:
        pass

    async def tag(self, text: str) -> POSResult:
        """
        Returns a list of tuples (token, tag). Example:
        [('I', 'PRON'), ('am', 'AUX'), ('30', 'NUM'), ('years', 'NOUN'), ('old', 'ADJ'), ('.', 'PUNCT')]
        """
        return await self._do_tag(text)
