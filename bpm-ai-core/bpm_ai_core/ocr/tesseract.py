import logging
import os
import urllib

from PIL import Image
from typing_extensions import override

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.ocr.ocr import OCR, OCRResult, OCRPage
from bpm_ai_core.util.image import pdf_to_images
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

    @override
    async def _do_process(
            self,
            blob: Blob,
            language: str = None
    ) -> OCRResult:
        if blob.is_pdf():
            images = pdf_to_images(await blob.as_bytes())
        elif blob.is_image():
            images = [Image.open(await blob.as_bytes_io())]
        else:
            raise ValueError("Blob must be a PDF or an image")
        if language is None:
            language = self.identify_image_language(images[0])
            logger.info(f"tesseract: auto detected language '{language}'")
        self.download_if_missing(language)

        pages = []
        for image in images:
            text = pytesseract.image_to_string(image)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            bboxes = []
            words = []

            # For each word in the data...
            for i in range(len(data['text'])):
                # ...if the word isn't empty
                if data['text'][i].strip():
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    # add bounding box
                    bboxes.append((x / image.width, y / image.height, (x + w) / image.width, (y + h) / image.height))
                    words.append(data['text'][i])

            page = OCRPage(
                text=text,
                words=words,
                bboxes=bboxes
            )
            pages.append(page)

        return OCRResult(pages=pages)

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
