import base64
import io
import logging
import tempfile
from io import BytesIO
from typing import Union, Tuple

from PIL import Image, ImageDraw
from pdf2image import convert_from_path, convert_from_bytes

logger = logging.getLogger(__name__)


image_ext_map = {
    'bmp': 'image/bmp',
    'gif': 'image/gif',
    'icns': 'image/x-icns',
    'ico': 'image/x-icon',
    'jfif': 'image/jpeg',
    'jpe': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'jpg': 'image/jpeg',
    'png': 'image/png',
    'pbm': 'image/x-portable-bitmap',
    'pgm': 'image/x-portable-graymap',
    'pnm': 'image/x-portable-anymap',
    'ppm': 'image/x-portable-pixmap',
    'tif': 'image/tiff',
    'tiff': 'image/tiff',
    'webp': 'image/webp',
}

pdf_ext_map = {
    'pdf': 'application/pdf'
}


async def blob_as_images(blob, accept_formats: list[str], return_bytes: bool = False) -> Union[list[Image.Image], list[bytes]]:
    """
    Load an image, PDF, or other file in a Blob into a Pillow Image object or raw bytes of accepted format.

    Args:
        blob: The input Blob object containing the image data.
        accept_formats: A list of accepted image formats (e.g., ['png', 'jpeg']).
        return_bytes: If True, return the image data as bytes instead of PIL Image objects.

    Returns:
        A list of PIL Image objects or a list of bytes representing the converted images.
    """
    if blob.is_pdf():
        # Convert PDF to a list of images
        logger.info("Converting PDF to a list of images...")
        images = pdf_to_images(await blob.as_bytes())
    elif blob.is_image():
        # Load the image from the blob
        images = [Image.open(await blob.as_bytes_io())]
    else:
        raise ValueError(f"Unsupported blob type: {blob.mimetype}")

    accept_formats = [f.lower() for f in accept_formats]

    # Convert images to the accepted formats if necessary
    converted_images = []
    for image in images:
        if image.format.lower() not in accept_formats:
            # Convert the image to the first accepted format
            output_format = accept_formats[0]
            logger.info(f"Converting images from {image.format.lower()} to accepted format: {output_format}")
            if return_bytes:
                # Convert the image to bytes
                with io.BytesIO() as output_bytes:
                    if not image.mode == "RGB":
                        image = image.convert("RGB")
                    image.save(output_bytes, format=output_format)
                    converted_images.append(output_bytes.getvalue())
            else:
                # Convert the image to PIL Image object
                converted_image = io.BytesIO()
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                image.save(converted_image, format=output_format)
                converted_image.seek(0)
                converted_images.append(Image.open(converted_image))
        else:
            if return_bytes:
                # Convert the image to bytes
                with io.BytesIO() as output_bytes:
                    image.save(output_bytes, format=image.format)
                    converted_images.append(output_bytes.getvalue())
            else:
                converted_images.append(image)

    return converted_images


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


def draw_boxes_on_image(image: Image, normalized_boxes: list[Tuple[float, float, float, float]]):
    """
    Draws bounding boxes on a given PIL image and displays the resulting image.

    Args:
        image (PIL.Image): The input image.
        normalized_boxes (list): A list of normalized bounding box coordinates.
                                 Each box is represented as a tuple (x1, y1, x2, y2),
                                 where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
                                 The coordinates are normalized, ranging from 0 to 1.

    Returns:
        None
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    # Iterate over the normalized bounding boxes
    for box in normalized_boxes:
        x1, y1, x2, y2 = box
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        # Draw the bounding box rectangle
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
    # Display the image with bounding boxes
    image.show()