from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ToolCallsMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.ocr.ocr import OCR
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.tracing.decorators import trace

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.common.json_utils import json_to_md
from bpm_ai.common.multimodal import transcribe_audio, prepare_images_for_llm_prompt, ocr_images


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

    store_task_result_tool = Tool.from_callable(
        "store_task_result",
        "Stores the result of the task.",
        args_schema=output_schema,
        callable=lambda **x: x
    )

    if llm.supports_images():
        input_data = prepare_images_for_llm_prompt(input_data)
    else:
        input_data = await ocr_images(input_data, ocr)
    input_data = await transcribe_audio(input_data, asr)

    input_md = json_to_md(input_data).strip()

    prompt = Prompt.from_file(
        "generic",
        context=input_md,
        task=instructions,
    )

    result = await llm.predict(prompt, tools=[store_task_result_tool])

    if isinstance(result, ToolCallsMessage):
        return result.tool_calls[0].invoke()
    else:
        return {}



