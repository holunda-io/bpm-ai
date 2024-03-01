from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ToolCallsMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.ocr.ocr import OCR
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.tracing.decorators import trace
from bpm_ai_core.translation.nmt import NMTModel

from bpm_ai.common.errors import BpmAiError, MissingParameterError, LanguageNotFoundError
from bpm_ai.common.multimodal import transcribe_audio, prepare_images_for_llm_prompt, ocr_images
from bpm_ai.translate.schema import get_translation_output_schema


@trace("bpm-ai-translate", ["llm"])
async def translate_llm(
        llm: LLM,
        input_data: dict[str, str | dict | None],
        target_language: str,
        ocr: OCR | None = None,
        asr: ASRModel | None = None
) -> dict:
    input_items = {k: v for k, v in input_data.items() if v is not None}
    if not input_items:
        return input_data

    if not target_language or target_language.isspace():
        raise MissingParameterError("target language is required")

    store_translation_tool = Tool.from_callable(
        "store_translation",
        f"Stores the finished translation into {target_language}.",
        args_schema=get_translation_output_schema(input_items, target_language),
        callable=lambda **x: x
    )

    if llm.supports_images():
        input_items = prepare_images_for_llm_prompt(input_items)
    else:
        input_items = await ocr_images(input_items, ocr)
    input_items = await transcribe_audio(input_items, asr)

    prompt = Prompt.from_file(
        "translate",
        input=input_items,
        lang=target_language
    )

    result = await llm.predict(prompt, tools=[store_translation_tool])

    if isinstance(result, ToolCallsMessage):
        result_items = result.tool_calls[0].invoke()
        return {k: result_items.get(k, None) for k in input_data.keys()}
    else:
        return {}


@trace("bpm-ai-translate", ["nmt"])
async def translate_nmt(
        nmt: NMTModel,
        input_data: dict[str, str | dict | None],
        target_language: str,
        ocr: OCR | None = None,
        asr: ASRModel | None = None
) -> dict:
    input_items = {k: v for k, v in input_data.items() if v is not None}
    if not input_items:
        return input_data

    if not target_language or target_language.isspace():
        raise MissingParameterError("target language is required")

    input_items = await ocr_images(input_items, ocr)
    input_items = await transcribe_audio(input_items, asr)

    try:
        import langcodes
        target_language_code = langcodes.find(target_language).language
    except ImportError:
        raise ImportError('langcodes is not installed')
    except LookupError:
        raise LanguageNotFoundError(f"Could not identify target language '{target_language}'.")

    texts_to_translate = list(input_items.values())
    texts_translated = nmt.translate(texts_to_translate, target_language_code)
    input_items_translated = {k: texts_translated[i] for i, k in enumerate(input_items.keys())}

    return {k: input_items_translated.get(k, None) for k in input_data.keys()}
