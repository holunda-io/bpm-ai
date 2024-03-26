import asyncio
from typing import List

from typing_extensions import override

from bpm_ai_core.translation.nmt import NMTModel

try:
    from aiobotocore.session import get_session

    has_amazon_translate = True
except ImportError:
    has_amazon_translate = False


class AmazonTranslate(NMTModel):
    """Amazon Translate NMT Model"""
    def __init__(self, region_name: str = None):
        if not has_amazon_translate:
            raise ImportError("aiobotocore is not installed")
        self.region_name = region_name

    async def _translate_single(self, text: str, target_language: str) -> str:
        async with get_session().create_client('translate', region_name=self.region_name) as client:
            response = await client.translate_text(
                Text=text,
                SourceLanguageCode='auto',
                TargetLanguageCode=target_language
            )
            return response['TranslatedText']

    @override
    async def _do_translate(self, text: str | List[str], target_language: str) -> str | List[str]:
        if isinstance(text, str):
            return await self._translate_single(text, target_language)
        else:
            tasks = [self._translate_single(t, target_language) for t in text]
            return await asyncio.gather(*tasks)
