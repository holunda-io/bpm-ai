import logging
import os
import urllib

from PIL.Image import Image

from bpm_ai_core.ocr.ocr import OCR, OCRResult
from bpm_ai_core.util.language import indentify_language_iso_639_3

try:
    import pytesseract
    has_pytesseract = True
except ImportError:
    has_pytesseract = False

logger = logging.getLogger(__name__)

TESSDATA_DIR = "~/.bpm.ai/tessdata/"


class TesseractOCR(OCR):
    """
    Local OCR model based on tesseract.

    To use, you should have the ``tesseract`` or ``tesseract-ocr`` package installed.
    """
    def __init__(self):
        if not has_pytesseract:
            raise ImportError('pytesseract is not installed')
        os.makedirs(TESSDATA_DIR, exist_ok=True)
        os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

    async def images_to_text_with_metadata(
            self,
            images: list[Image],
            language: str = None
    ) -> OCRResult:
        if language is None:
            language = self.identify_image_language(images[0])
            logger.info(f"tesseract: auto detected language '{language}'")
        self.download_if_missing(language)
        texts = [pytesseract.image_to_string(image, lang=language) for image in images]
        return OCRResult(texts=texts)

    def identify_image_language(self, image: Image) -> str:
        self.download_if_missing('eng')
        text = pytesseract.image_to_string(image)
        return indentify_language_iso_639_3(text)

    @staticmethod
    def download_if_missing(lang: str):
        lang_file = f'{lang}.traineddata'
        tessdata_file_path = os.path.join(TESSDATA_DIR, lang_file)
        if not os.path.exists(tessdata_file_path):
            logger.info(f'tesseract: {lang_file} not found in {TESSDATA_DIR}, downloading...')
            download_url = f'https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang_file}'
            urllib.request.urlretrieve(download_url, tessdata_file_path)
            logger.info(f'tesseract: Downloaded {lang_file} to tessdata directory')
