from bpm_ai_core.ocr.tesseract import TesseractOCR
from bpm_ai_core.util.image import load_images


async def test_tesseract_image():
    ocr = TesseractOCR()

    image = load_images("example.png")
    text = await ocr.images_to_text(image)

    assert "example" in text


async def test_tesseract_pdf():
    ocr = TesseractOCR()

    image = load_images("dummy.pdf")
    text = await ocr.images_to_text(image)

    assert "Dummy PDF file" in text
