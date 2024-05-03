from typing import Any

from bpm_ai_core.image_classification.image_classifier import ImageClassifier
from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.ocr.ocr import OCR
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.text_classification.text_classifier import TextClassifier
from bpm_ai_core.tracing.decorators import trace
from bpm_ai_core.util.file import is_supported_img_file
from bpm_ai_core.util.markdown import dict_to_md

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.common.multimodal import transcribe_audio, prepare_images_for_llm_prompt, ocr_documents, prepare_text_blobs, \
    assert_all_files_processed, replace_text_blobs
from bpm_ai.decide.schema import get_cot_decision_output_schema, get_decision_output_schema, remove_order_prefix_from_keys


@trace("bpm-ai-decide", ["llm"])
async def decide_llm(
    llm: LLM,
    input_data: dict[str, str | dict | None],
    instructions: str,
    output_type: str,
    possible_values: list[Any] | None = None,
    multiple_decision_values: bool = False,
    strategy: str | None = None,
    ocr: OCR | None = None,
    asr: ASRModel | None = None
) -> dict:
    if not instructions or instructions.isspace():
        raise MissingParameterError("question/instruction is required")
    if not output_type or output_type.isspace():
        raise MissingParameterError("output type is required")

    if all(value is None for value in input_data.values()):
        return {"decision": None, "reasoning": "No input values present."}

    if strategy == 'cot':
        output_schema = get_cot_decision_output_schema(output_type, possible_values, multiple_decision_values)
    else:
        output_schema = get_decision_output_schema(output_type, possible_values, multiple_decision_values)

    if not ocr and llm.supports_images():
        input_data = prepare_images_for_llm_prompt(input_data)
    else:
        input_data = await ocr_documents(input_data, ocr)
    input_data = await transcribe_audio(input_data, asr)
    input_data = prepare_text_blobs(input_data)
    assert_all_files_processed(input_data)

    prompt = Prompt.from_file(
        "decide",
        context=input_data,
        task=instructions,
        output_type=output_type,
        possible_values=possible_values,
        multiple_decision_values=multiple_decision_values,
        strategy=strategy
    )

    decide_schema = {
        "name": "store_decision",
        "description": f"Stores the final decision value{'s' if multiple_decision_values else ''} and corresponding reasoning.",
        "type": "object",
        "properties": output_schema
    }

    message = await llm.generate_message(prompt, output_schema=decide_schema)

    return remove_order_prefix_from_keys(message.content) if message.content else {}


@trace("bpm-ai-decide", ["classifier"])
async def decide_classifier(
    input_data: dict[str, str | dict | None],
    output_type: str,
    classifier: TextClassifier = None,
    image_classifier: ImageClassifier = None,
    question: str | None = None,
    possible_values: list[Any] | None = None,
    multiple_decision_values: bool = False,
    ocr: OCR | None = None,
    asr: ASRModel | None = None
) -> dict:
    if (classifier and image_classifier) or not (classifier or image_classifier):
        raise ValueError("Must provide either a TextClassifier or an ImageClassifier")
    if image_classifier and multiple_decision_values:
        raise ValueError("ImageClassifier does not support multiple decision values")

    zero_shot = (possible_values is not None) or output_type == "boolean"

    if not output_type or output_type.isspace():
        raise MissingParameterError("output type is required")
    if zero_shot and output_type == "boolean":
        possible_values = ["yes", "no"]
    if zero_shot:
        possible_values = [str(v) for v in possible_values]

    if all(value is None for value in input_data.values()):
        return {"decision": None, "reasoning": "No input values present."}

    if image_classifier and not classifier:
        image_path_or_url = list(input_data.values())[0]
        if len(input_data) > 1 or not isinstance(image_path_or_url, str) or not is_supported_img_file(image_path_or_url):
            raise ValueError("Must provide exactly one image file path variable as input when using an Image Classifier.")

        hypothesis_template = "Based on this image, the question '" + question + "' should be answered with '{}'" \
            if question else "This is an image of {}."

        classification = await image_classifier.classify(
            blob_or_path=image_path_or_url,
            classes=possible_values,
            hypothesis_template=hypothesis_template,
            confidence_threshold=0.1,
        )

    else:  # text classifier
        input_data = await ocr_documents(input_data, ocr)
        input_data = await transcribe_audio(input_data, asr)
        input_data = await replace_text_blobs(input_data)
        assert_all_files_processed(input_data)

        input_md = dict_to_md(input_data).strip()

        hypothesis_template = "In this example the question '" + question + "' should be answered with '{}'" \
            if question else "This example is {}."

        classification = await classifier.classify(
            input_md,
            possible_values,
            hypothesis_template=hypothesis_template,
            confidence_threshold=0.1,
            multi_label=multiple_decision_values
        )

    def raw_to_output_type(raw: str) -> Any:
        if output_type == "boolean":
            return (raw.lower() == 'yes' or raw.lower() == "true") if raw else None
        elif output_type == "integer":
            return int(raw) if raw else None
        elif output_type == "number":
            return float(raw) if raw else None
        else:
            return raw

    if not classification:
        result = None
    elif multiple_decision_values:
        result = [raw_to_output_type(label) for label, _ in classification.labels_scores]
    else:
        result = raw_to_output_type(classification.max_label)

    return {
        "decision": result,
        "reasoning": ""
    }
