from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.util.image import blob_as_images


async def test_blob_to_image_no_conversion():
    blob = Blob.from_path_or_url('example.png')
    images = await blob_as_images(blob, accept_formats=["jpeg", "png"])

    assert images[0].format == "PNG"

    blob = Blob.from_path_or_url('example.jpg')
    images = await blob_as_images(blob, accept_formats=["jpeg", "png"])

    assert images[0].format == "JPEG"


async def test_blob_to_image_conversion():
    blob = Blob.from_path_or_url('example.png')
    images = await blob_as_images(blob, accept_formats=["jpeg"])
    assert images[0].format == "JPEG"

    blob = Blob.from_path_or_url('example.jpg')
    images = await blob_as_images(blob, accept_formats=["png"])
    assert images[0].format == "PNG"

    blob = Blob.from_path_or_url('sample-invoice.webp')
    images = await blob_as_images(blob, accept_formats=["jpeg"])
    assert images[0].format == "JPEG"

    blob = Blob.from_path_or_url('invoice-sample.pdf')
    images = await blob_as_images(blob, accept_formats=["jpeg"])
    assert images[0].format == "JPEG"
