from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.ocr.ocr import OCR
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.tracing.decorators import trace

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.common.multimodal import transcribe_audio, prepare_images_for_llm_prompt, ocr_documents


@trace("bpm-ai-generic", ["llm"])
async def generic_llm(
    llm: LLM,
    input_data: dict[str, str | dict],
    instructions: str,
    output_schema: dict[str, str | dict],
    ocr: OCR | None = None,
    asr: ASRModel | None = None
) -> dict:
    if not instructions or instructions.isspace():
        raise MissingParameterError("instructions are required")
    if not output_schema:
        raise MissingParameterError("output schema is required")

    if not ocr and llm.supports_images():
        input_data = prepare_images_for_llm_prompt(input_data)
    else:
        input_data = await ocr_documents(input_data, ocr)
    input_data = await transcribe_audio(input_data, asr)

    prompt = Prompt.from_file(
        "generic",
        context=input_data,
        task=instructions,
    )

    generic_schema = {
        "name": "store_task_result",
        "description": "Stores the result of the task.",
        "type": "object",
        "properties": output_schema
    }

    message = await llm.generate_message(prompt, output_schema=generic_schema)

    return message.content or {}



