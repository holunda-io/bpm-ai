from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.ocr.ocr import OCR
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.util.file import is_supported_audio_file, is_file, is_supported_text_file
from bpm_ai_core.util.file import is_supported_img_file

from bpm_ai.common.errors import FileNotSupportedError


def prepare_images_for_llm_prompt(input_data: dict):
    """
    For multi-modal LLMs. Will be turned into Blob objects as part of the prompt processing
    and then into the respective image format of the LLM.
    """
    return {
        k: f"[# blob {v} #]"
        if (isinstance(v, str) and is_supported_img_file(v))
        else v for k, v in input_data.items()
    }


async def ocr_documents(input_data: dict, ocr: OCR | None = None):
    return {
        k: (await ocr.process(v)).full_text
        if (ocr and isinstance(v, str) and is_supported_img_file(v))
        else v for k, v in input_data.items()
    }


async def transcribe_audio(input_data: dict, asr: ASRModel | None = None) -> dict:
    return {
        k: (await asr.transcribe(v)).text
        if (asr and isinstance(v, str) and is_supported_audio_file(v))
        else v for k, v in input_data.items()
    }


def prepare_text_blobs(input_data: dict):
    """
    Will be turned into Blob objects as part of the prompt processing and then into text.
    """
    return {
        k: f"[# blob {v} #]"
        if (isinstance(v, str) and is_supported_text_file(v))
        else v for k, v in input_data.items()
    }


async def replace_text_blobs(input_data: dict):
    return {
        k: (await Blob.from_path_or_url(v).as_bytes()).decode("utf-8")
        if (isinstance(v, str) and is_supported_text_file(v))
        else v for k, v in input_data.items()
    }


def assert_all_files_processed(input_data: dict):
    for v in input_data.values():
        if v and isinstance(v, str) and is_file(v):
            raise FileNotSupportedError(v)
