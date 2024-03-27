import io
import os

import requests


audio_ext_map = {
    'flac': 'audio/flac',
    'mp3': 'audio/mpeg',
    'mp4': 'audio/mpeg',
    'mpeg': 'audio/mpeg',
    'mpga': 'audio/mpeg',
    'm4a': 'audio/mpeg',
    'ogg': 'audio/ogg',
    'wav': 'audio/vnd.wav',
    'webm': 'audio/webm',
}


def load_audio(path: str) -> io.BytesIO:
    """
    Load an audio file from a local path or a web URL into a BytesIO object.

    Parameters:
    - path (str): A file system path or a URL of an audio file.

    Returns:
    - BytesIO: A BytesIO object containing the audio data.
    """
    if path.startswith('http://') or path.startswith('https://'):
        # Handle web URL
        response = requests.get(path)
        audio = io.BytesIO(response.content)
        if path.endswith(tuple(audio_ext_map.keys())):
            audio.name = f"audio.{path.rsplit('.', 1)[-1]}"
    elif os.path.isfile(path):
        # Handle local file path
        with open(path, 'rb') as f:
            audio = io.BytesIO(f.read())
            audio.name = f.name
    else:
        raise ValueError("The path provided is neither a valid URL nor a file path.")

    return audio