from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ToolCallsMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.tracing.decorators import trace
from bpm_ai_core.translation.nmt import NMTModel

from bpm_ai.common.multimodal import prepare_audio
from bpm_ai.translate.schema import get_translation_output_schema


@trace("bpm-ai-translate", ["llm"])
async def translate_llm(
    llm: LLM,
    input_data: dict[str, str | dict],
    target_language: str,
    asr: ASRModel | None = None
) -> dict:
    tool = Tool.from_callable(
        "store_translation",
        f"Stores the finished translation into {target_language}.",
        args_schema=get_translation_output_schema(input_data, target_language),
        callable=lambda **x: x
    )

    #input_data = prepare_images(input_data)  todo enable once GPT-4V is stable
    input_data = await prepare_audio(input_data, asr)

    prompt = Prompt.from_file(
        "translate",
        input=input_data,
        lang=target_language
    )

    result = await llm.predict(prompt, tools=[tool])

    if isinstance(result, ToolCallsMessage):
        return result.tool_calls[0].invoke()
    else:
        return {}


@trace("bpm-ai-translate", ["nmt"])
async def translate_nmt(
    nmt: NMTModel,
    input_data: dict[str, str | dict],
    target_language: str,
    asr: ASRModel | None = None
) -> dict:
    #input_data = prepare_images(input_data)  todo
    input_data = await prepare_audio(input_data, asr)

    try:
        import langcodes
        target_language_code = langcodes.find(target_language).language
    except ImportError:
        raise ImportError('langcodes is not installed')
    except LookupError:
        raise Exception(f"Could not identify target language '{target_language}'.")

    texts_to_translate = list(input_data.values())  # todo handle None values
    translated_texts = nmt.translate(texts_to_translate, target_language_code)

    return {k: translated_texts[i] for i, k in enumerate(input_data.keys())}

