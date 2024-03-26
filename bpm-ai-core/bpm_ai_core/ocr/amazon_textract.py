import asyncio

from typing_extensions import override

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.ocr.ocr import OCR, OCRResult, OCRPage
from bpm_ai_core.util.image import blob_as_images
from bpm_ai_core.util.storage import is_s3_url, parse_s3_url

try:
    from aiobotocore.session import get_session
    from textractprettyprinter.t_pretty_print import get_text_from_layout_json

    has_textract = True
except ImportError:
    has_textract = False

IMAGE_FORMATS = ["png", "jpeg", "tiff"]


class AmazonTextractOCR(OCR):
    def __init__(self, region_name: str = None):
        if not has_textract:
            raise ImportError('aiobotocore and/or amazon-textract-prettyprinter are not installed')
        self.region_name = region_name

    @override
    async def _do_process(
        self,
        blob: Blob,
        language: str = None
    ) -> OCRResult:
        if not (blob.is_pdf() or blob.is_image()):
            raise ValueError("Blob must be a PDF or an image")

        if is_s3_url(blob.path):
            pages = await self._get_pages_async(blob.path)
        else:
            pages = await self._get_pages_sync(blob)

        return OCRResult(pages=pages)

    async def _get_pages_sync(self, document: Blob):
        if document.is_pdf():
            _bytes = await document.as_bytes()
        else:
            _bytes = (await blob_as_images(document, accept_formats=IMAGE_FORMATS, return_bytes=True))[0]
        # Create a document from the image bytes asynchronously
        async with get_session().create_client("textract", region_name=self.region_name) as client:
            # Call Amazon Textract API asynchronously
            response = await client.analyze_document(
                Document={"Bytes": _bytes},
                FeatureTypes=["TABLES", "FORMS", "LAYOUT"]
            )
        # Convert Textract response to markdown using amazon-textract-prettyprinter
        markdown_pages = get_text_from_layout_json(
            textract_json=response,
            table_format="github",
            generate_markdown=True
        )
        return self.parse_pages(markdown_pages, response)

    async def _get_pages_async(self, s3_url: str):
        bucket_name, file_path = await parse_s3_url(s3_url)
        # Create a document from the image bytes asynchronously
        async with get_session().create_client("textract", region_name=self.region_name) as client:
            # Call Amazon Textract API asynchronously using start_document_analysis
            response = await client.start_document_analysis(
                DocumentLocation={'S3Object': {
                    'Bucket': bucket_name,
                    'Name': file_path
                }},
                FeatureTypes=["TABLES", "FORMS", "LAYOUT"]
            )

            # Get the job ID from the response
            job_id = response["JobId"]

            # Wait for the job to complete
            while True:
                response = await client.get_document_analysis(JobId=job_id)
                status = response["JobStatus"]
                if status in ["SUCCEEDED", "FAILED"]:
                    break
                await asyncio.sleep(1)  # Wait for 1 second before checking the status again

            if status == "FAILED":
                raise Exception(f"Document analysis failed with error: {response['StatusMessage']}")

            # Retrieve the results from the completed job
            pages = []
            markdown_pages = {}
            next_token = None
            while True:
                if next_token:
                    response = await client.get_document_analysis(JobId=job_id, NextToken=next_token)
                else:
                    response = await client.get_document_analysis(JobId=job_id)

                # Convert Textract response to markdown using amazon-textract-prettyprinter
                markdown_pages.update(get_text_from_layout_json(
                    textract_json=response,
                    table_format="github",
                    generate_markdown=True
                ))
                pages.extend(self.parse_pages(markdown_pages, response))

                next_token = response.get("NextToken")
                if not next_token:
                    break
            return pages

    @staticmethod
    def parse_pages(markdown_pages, response):
        pages = []
        page_idx = -1
        for block in response["Blocks"]:
            if block["BlockType"] == "PAGE":
                page_idx += 1
                bboxes = []
                words = []
                for _block in block["Relationships"][0]["Ids"]:
                    block_data = next(b for b in response["Blocks"] if b["Id"] == _block)
                    if block_data["BlockType"] == "LINE":
                        for word_block_id in block_data["Relationships"][0]["Ids"]:
                            word_block = next(b for b in response["Blocks"] if b["Id"] == word_block_id)
                            if word_block["BlockType"] == "WORD":
                                bbox = word_block["Geometry"]["BoundingBox"]
                                x, y, w, h = bbox["Left"], bbox["Top"], bbox["Width"], bbox["Height"]
                                bboxes.append((x, y, x + w, y + h))
                                words.append(word_block["Text"])
                page_data = OCRPage(
                    text=list(markdown_pages.values())[page_idx],
                    words=words,
                    bboxes=bboxes
                )
                pages.append(page_data)
        return pages
