from typing import Any

from bpm_ai_core.classification.zero_shot_classifier import ZeroShotClassifier
from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ToolCallsMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.tracing.decorators import trace

from bpm_ai.common.json_utils import json_to_md
from bpm_ai.common.multimodal import prepare_audio
from bpm_ai.decide.schema import get_cot_decision_output_schema, get_decision_output_schema


@trace("bpm-ai-decide", ["llm"])
async def decide_llm(
    llm: LLM,
    input_data: dict[str, str | dict],
    instructions: str,
    output_type: str,
    possible_values: list[Any] | None = None,
    strategy: str | None = None,
    asr: ASRModel | None = None
) -> dict:
    if strategy == 'cot':
        output_schema = get_cot_decision_output_schema(output_type, possible_values)
    else:
        output_schema = get_decision_output_schema(output_type, possible_values)

    tool = Tool.from_callable(
        "store_decision",
        "Stores the final decision value and corresponding reasoning.",
        args_schema=output_schema,
        callable=lambda **x: x
    )

    #input_data = prepare_images(input_data)  todo enable once GPT-4V is stable
    input_data = prepare_audio(input_data, asr)

    input_md = json_to_md(input_data).strip()

    prompt = Prompt.from_file(
        "decide",
        context=input_md,
        task=instructions,
        output_type=output_type,
        possible_values=possible_values,
        strategy=strategy
    )

    result = await llm.predict(prompt, tools=[tool])

    if isinstance(result, ToolCallsMessage):
        return result.tool_calls[0].invoke()
    else:
        return {}


@trace("bpm-ai-decide", ["classifier"])
async def decide_classifier(
    classifier: ZeroShotClassifier,
    input_data: dict[str, str | dict],
    output_type: str,
    question: str | None = None,
    possible_values: list[Any] | None = None,
    asr: ASRModel | None = None
) -> dict:
    if not possible_values and output_type != "boolean":
        raise Exception("List of possible values must be specified for classifier (except boolean)")
    if output_type == "boolean":
        possible_values = ["yes", "no"]
    possible_values = [str(v) for v in possible_values]

    #input_data = prepare_images(input_data)  todo enable once GPT-4V is stable
    input_data = prepare_audio(input_data, asr)

    input_md = json_to_md(input_data).strip()

    hypothesis_template = "In this example the question '" + question + "' should be answered with '{}'" \
        if question else "This example is {}."

    result_raw = classifier.classify(
        input_md,
        possible_values,
        hypothesis_template=hypothesis_template,
        confidence_threshold=0.1
    )

    if output_type == "boolean":
        result = (result_raw == 'yes') if result_raw else None
    elif output_type == "integer":
        result = int(result_raw) if result_raw else None
    elif output_type == "float":
        result = float(result_raw) if result_raw else None
    else:
        result = result_raw

    return {"decision": result}
