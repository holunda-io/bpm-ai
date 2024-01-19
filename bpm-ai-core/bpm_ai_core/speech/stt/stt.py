import io
from abc import ABC, abstractmethod
from typing import Optional, Union


class STTModel(ABC):

    @abstractmethod
    def transcribe(self, audio: Union[str, io.BytesIO], language: Optional[str] = None) -> str:
        pass
