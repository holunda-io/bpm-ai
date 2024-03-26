import logging
import os
from io import BytesIO
from typing_extensions import override

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.ocr.ocr import OCR, OCRResult, OCRPage
from bpm_ai_core.util.image import blob_as_images

try:
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient as AsyncDocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    has_azure_doc = True
except ImportError:
    has_azure_doc = False

azure_logger = logging.getLogger('azure')
azure_logger.setLevel(logging.WARNING)

IMAGE_FORMATS = ["png", "jpeg", "tiff"]


class AzureOCR(OCR):
    def __init__(self, endpoint: str = None):
        if not has_azure_doc:
            raise ImportError('azure-ai-documentintelligence is not installed')
        self.endpoint = endpoint or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

    @override
    async def _do_process(
        self,
        blob: Blob,
        language: str = None
    ) -> OCRResult:
        if not (blob.is_pdf() or blob.is_image()):
            raise ValueError("Blob must be a PDF or an image")
        if blob.is_pdf():
            bytes_io = await blob.as_bytes_io()
        else:
            bytes_io = BytesIO((await blob_as_images(blob, accept_formats=IMAGE_FORMATS, return_bytes=True))[0])

        async with AsyncDocumentIntelligenceClient(
                self.endpoint,
                AzureKeyCredential(
                    os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
                )
        ) as client:
            document = await client.begin_analyze_document(
                model_id="prebuilt-layout",
                analyze_request=bytes_io,
                content_type="application/octet-stream",
                output_content_format="markdown"
            )

            # Wait for the extraction to complete asynchronously
            result = await document.result()
            markdown_content = result.content

            pages = []
            for page in result.pages:
                bboxes = []
                words = []
                for word in page.words:
                    polygon = word.polygon
                    x, y = polygon[0], polygon[1]
                    w, h = polygon[2] - x, polygon[5] - y
                    bboxes.append((x / page['width'], y / page['height'], (x + w) / page['width'], (y + h) / page['height']))
                    words.append(word.content)

                page_data = OCRPage(
                    text=" ".join(words),
                    words=words,
                    bboxes=bboxes
                )
                pages.append(page_data)

        return OCRResult(pages=pages)
