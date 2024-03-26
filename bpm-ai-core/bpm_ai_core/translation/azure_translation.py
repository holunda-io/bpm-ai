import logging
import os
from typing_extensions import override

from bpm_ai_core.translation.nmt import NMTModel

try:
    from azure.ai.translation.text import TranslatorCredential
    from azure.ai.translation.text.aio import TextTranslationClient
    from azure.ai.translation.text.models import InputTextItem

    has_azure_translation = True
except ImportError:
    has_azure_translation = False

azure_logger = logging.getLogger('azure')
azure_logger.setLevel(logging.WARNING)


class AzureTranslation(NMTModel):
    """Azure Text Translation NMT Model"""

    def __init__(self, endpoint: str = None, region: str = "westeurope"):
        if not has_azure_translation:
            raise ImportError('azure-ai-translation-text  is not installed')
        self.key = os.getenv('AZURE_TRANSLATION_KEY')
        self.endpoint = endpoint or os.getenv('AZURE_TRANSLATION_ENDPOINT')
        self.region = region or os.getenv('AZURE_TRANSLATION_REGION')

    @override
    async def _do_translate(self, text: str | list[str], target_language: str) -> str | list[str]:
        async with TextTranslationClient(
                endpoint=self.endpoint,
                credential=TranslatorCredential(self.key, self.region)
        ) as client:
            if isinstance(text, str):
                texts = [text]
            else:
                texts = text
            items = [InputTextItem(text=text) for text in texts]
            response = await client.translate(content=items, to=[target_language])
            return [doc.translations[0].text for doc in response]
