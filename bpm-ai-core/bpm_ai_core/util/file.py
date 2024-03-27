import os
from typing import List
from urllib.parse import urlparse

from bpm_ai_core.util.audio import audio_ext_map
from bpm_ai_core.util.image import image_ext_map, pdf_ext_map

supported_ext_map = {**audio_ext_map, **image_ext_map, **pdf_ext_map}
supported_extensions = supported_ext_map.keys()


def guess_mimetype(filename: str) -> str | None:
    if is_supported_file(filename, list(image_ext_map.keys())):
        return image_ext_map[_extract_extension(filename)]
    elif is_supported_file(filename, list(pdf_ext_map.keys())):
        return "application/pdf"
    elif is_supported_file(filename, list(audio_ext_map.keys())):
        return audio_ext_map[_extract_extension(filename)]


def is_supported_file(url_or_path: str, extensions: List[str] = supported_extensions) -> bool:
    file_extension = _extract_extension(url_or_path)
    # Normalize the extensions to lowercase
    extensions = [ext.lower() for ext in extensions]
    # Check if the file extension is in the list of supported extensions
    return file_extension in extensions


def is_supported_img_file(url_or_path: str) -> bool:
    return is_supported_file(url_or_path, extensions=list(image_ext_map.keys()) + list(pdf_ext_map.keys()))


def is_supported_audio_file(url_or_path: str) -> bool:
    return is_supported_file(url_or_path, extensions=list(audio_ext_map.keys()))


def _extract_extension(url_or_path):
    url_or_path = url_or_path.strip()
    # Extract the path from URL if it's a URL
    parsed_url = urlparse(url_or_path)
    path = parsed_url.path if parsed_url.scheme else url_or_path
    # Remove trailing slash if present
    if path.endswith('/'):
        path = path[:-1]
    # Extract the file extension
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension.lower().lstrip('.')
    return file_extension
