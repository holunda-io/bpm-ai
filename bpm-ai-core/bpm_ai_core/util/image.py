import base64
import os
import tempfile
from io import BytesIO

import requests
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes

from bpm_ai_core.util.file import is_supported_file

supported_img_extensions = [
    'bmp', 'dib',
    'gif',
    'icns', 'ico',
    'jfif', 'jpe', 'jpeg', 'jpg',
    'j2c', 'j2k', 'jp2', 'jpc', 'jpf', 'jpx',
    'apng', 'png',
    'pbm', 'pgm', 'pnm', 'ppm',
    'tif', 'tiff',
    'webp',
    'emf', 'wmf',
    'pdf'
]


def is_supported_img_file(url_or_path: str) -> bool:
    return is_supported_file(url_or_path, supported_extensions=supported_img_extensions)


def load_images(path: str) -> list[Image]:
    """
    Load an image or pdf from a local path or a web URL into Pillow Image objects.

    Parameters:
    - path (str): A file system path or a URL of an image or pdf.

    Returns:
    - list[Image]: A list of PIL Images (single element for normal images, or an image for each pdf page).
    """
    if path.startswith('http://') or path.startswith('https://'):
        # Handle web URL
        response = requests.get(path)
        try:
            images = [Image.open(BytesIO(response.content))]
        except Exception:
            images = pdf_to_images(response.content)
    elif os.path.isfile(path):
        # Handle local file path
        try:
            images = [Image.open(path)]
        except Exception:
            images = pdf_to_images(path)
    else:
        raise ValueError("The path provided is neither a valid URL nor a file path.")

    return images


def pdf_to_images(pdf: bytes | str) -> list[Image]:
    with tempfile.TemporaryDirectory() as path:
        if isinstance(pdf, bytes):
            func = convert_from_bytes
        else:
            func = convert_from_path
        return func(pdf, output_folder=path, dpi=100, use_pdftocairo=True)


def base64_encode_image(image: Image):
    """
    Get a base64 encoded string from a Pillow Image object.

    Parameters:
    - image (Image): A Pillow Image object.

    Returns:
    - str: Base64 encoded string of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format=image.format or "JPEG")  # Assuming JPEG if format is not provided
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')
