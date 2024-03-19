import os

import pytest
import requests

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


async def test_tesseract_bbox():
    from PIL import Image, ImageDraw
    import pytesseract

    # Load the image from file
    image = load_images("https://slicedinvoices.com/pdf/wordpress-pdf-invoice-plugin-sample.pdf")[0]

    # Use Tesseract to do OCR on the image
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # For each word in the data,
    # draw a box around it on the image
    for i in range(len(data['text'])):
        # If the word isn't empty,
        if data['text'][i].strip():
            # Get the bounding box
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Draw the box
            draw.rectangle(((x, y), (x + w, y + h)), outline='red')

    # Save the image with the boxes
    image.save('output.jpg')


@pytest.mark.skip
async def test_azure_bbox():
    from azure.ai.formrecognizer import FormRecognizerClient
    from azure.core.credentials import AzureKeyCredential
    from PIL import Image, ImageDraw

    # Set up the FormRecognizerClient
    endpoint = "https://westeurope.api.cognitive.microsoft.com/"
    key = os.environ.get("AZURE_DOCUMENT_AI_KEY")
    form_recognizer_client = FormRecognizerClient(endpoint, AzureKeyCredential(key))

    # Load the image from file
    image_path = 'doc.png'
    with open(image_path, "rb") as fd:
        form = fd.read()

    # Use Azure Form Recognizer to do OCR on the image
    poller = form_recognizer_client.begin_recognize_content(form)
    result = poller.result()

    # Load the image again for drawing
    image = Image.open(image_path)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # For each line in the result,
    for page in result:
        for line in page.lines:
            # Get the bounding box
            x1, y1 = line.bounding_box[0]
            x2, y2 = line.bounding_box[2]
            # Draw the box
            draw.rectangle(((x1, y1), (x2, y2)), outline='red')

    # Save the image with the boxes
    image.save('output.png')