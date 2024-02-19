from bpm_ai_core.ocr.ocr import OCR
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.util.audio import is_supported_audio_file
from bpm_ai_core.util.image import is_supported_img_file


def prepare_images_for_llm_prompt(input_data: dict):
    """
    For multi-modal LLMs. Will be turned into PIL Image(s) as part of the prompt processing.
    """
    return {
        k: f"[# image {v} #]"
        if (isinstance(v, str) and is_supported_img_file(v))
        else v for k, v in input_data.items()
    }


async def ocr_images(input_data: dict, ocr: OCR | None = None):
    return {
        k: await ocr.images_to_text(v)
        if (ocr and isinstance(v, str) and is_supported_img_file(v))
        else v for k, v in input_data.items()
    }


async def transcribe_audio(input_data: dict, asr: ASRModel | None = None) -> dict:
    return {
        k: await asr.transcribe(v)
        if (asr and isinstance(v, str) and is_supported_audio_file(v))
        else v for k, v in input_data.items()
    }