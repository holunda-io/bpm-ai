from abc import ABC, abstractmethod
from typing import Tuple


class POSTagger(ABC):
    """
    Part-of-Speech Tagging Model
    """

    @abstractmethod
    def tag(self, text: str) -> list[Tuple[str, str]]:
        """
        Returns a list of tuples (token, tag). Example:
        [('I', 'PRON'), ('am', 'AUX'), ('30', 'NUM'), ('years', 'NOUN'), ('old', 'ADJ'), ('.', 'PUNCT')]
        """
        pass
