import os

import pytest
from PIL import Image

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.ocr.amazon_textract import AmazonTextractOCR
from bpm_ai_core.ocr.azure_doc_intelligence import AzureOCR
from bpm_ai_core.util.image import pdf_to_images, draw_boxes_on_image, pdf_to_images_poppler


async def test_pdf():
    images = pdf_to_images_poppler("files/invoice.pdf")
    print(images[0].format)



async def test_pdf_s3():
    ocr = AzureOCR(endpoint="https://westeurope.api.cognitive.microsoft.com/")

    result = await ocr.process("s3://bennet-test-bucket/Test_PDF.pdf")

    assert "Muster pdf Dokument" in result.full_text


async def test_azure_pdf():
    ocr = AzureOCR(endpoint="https://westeurope.api.cognitive.microsoft.com/")

    result = await ocr.process("files/invoice.pdf")

    print(result)
    #assert "INV-3337" in result.full_text


async def test_azure_image():
    ocr = AzureOCR(endpoint="https://westeurope.api.cognitive.microsoft.com/")

    result = await ocr.process("example.png")

    print(result)
    draw_boxes_on_image(Image.open("example.png"), result.pages[0].bboxes)
    assert "example image" in result.full_text


async def test_azure_image_unsupported_type():
    ocr = AzureOCR(endpoint="https://westeurope.api.cognitive.microsoft.com/")

    result = await ocr.process("sample-invoice.webp")

    print(result)
    assert "Editorial photo shoot in Yucatan, Mexico" in result.full_text


async def test_textract_image():
    ocr = AmazonTextractOCR()

    result = await ocr.process("example.png")

    assert "example image" in result.full_text

    draw_boxes_on_image(Image.open("example.png"), result.pages[0].bboxes)


async def test_textract_image_unsupported_type():
    ocr = AmazonTextractOCR()

    result = await ocr.process("sample-invoice.webp")

    print(result)
    assert "Editorial photo shoot in Yucatan, Mexico" in result.full_text


async def test_textract_pdf():
    ocr = AmazonTextractOCR()

    result = await ocr.process("s3://bennet-test-bucket/pdfs/dummy.pdf")

    print(result)
    assert "Dummy" in result.full_text
