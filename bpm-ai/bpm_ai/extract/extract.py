import re
from typing import Callable, Any

from bpm_ai_core.classification.zero_shot_classifier import ZeroShotClassifier
from bpm_ai_core.question_answering.question_answering import QuestionAnswering
from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.ocr.ocr import OCR
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.token_classification.zero_shot_token_classifier import ZeroShotTokenClassifier
from bpm_ai_core.tracing.decorators import trace
from bpm_ai_core.util.json_schema import expand_simplified_json_schema
from bpm_ai_core.util.markdown import dict_to_md

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.common.multimodal import transcribe_audio, prepare_images_for_llm_prompt, ocr_documents


@trace("bpm-ai-extract", ["llm"])
async def extract_llm(
    llm: LLM,
    input_data: dict[str, str | dict | None],
    output_schema: dict[str, str | dict],
    multiple: bool = False,
    multiple_description: str = "",
    ocr: OCR | None = None,
    asr: ASRModel | None = None
) -> dict | list[dict]:
    if all(value is None for value in input_data.values()):
        return input_data

    if not ocr and llm.supports_images():
        input_data = prepare_images_for_llm_prompt(input_data)
    else:
        input_data = await ocr_documents(input_data, ocr)
    input_data = await transcribe_audio(input_data, asr)

    if not output_schema:
        return input_data

    def transform_result(extracted: dict):
        def empty_to_none(v):
            return None if v in ["", "null"] else v

        if multiple and "entities" in extracted.keys():
            extracted = extracted["entities"]

        if isinstance(extracted, list):
            return [transform_result(d) for d in extracted]
        else:
            return {k: empty_to_none(v) for k, v in extracted.items()}

    prompt = Prompt.from_file("extract", input=input_data)

    extract_schema = {
        "name": "information_extraction",
        "description": f"Extracts the relevant {'entities' if multiple else 'information'} from the passage.",
        "type": "object",
        "properties": output_schema if not multiple else {
            "entities": {"type": "array", "description": multiple_description, "items": output_schema}
        }
    }

    message = await llm.generate_message(prompt, output_schema=extract_schema)

    return transform_result(message.content or {})


@trace("bpm-ai-extract", ["extractive-qa"])
async def extract_qa(
    qa: QuestionAnswering,
    classifier: ZeroShotClassifier,
    token_classifier: ZeroShotTokenClassifier,
    input_data: dict[str, str | dict | None],
    output_schema: dict[str, str | dict],
    multiple: bool = False,
    multiple_description: str = "",
    ocr: OCR | None = None,
    asr: ASRModel | None = None
) -> dict | list[dict]:
    if all(value is None for value in input_data.values()):
        return input_data

    input_data = await ocr_documents(input_data, ocr)
    input_data = await transcribe_audio(input_data, asr)

    if not output_schema:
        return input_data

    input_md = dict_to_md(input_data).strip()
    output_schema = expand_simplified_json_schema(output_schema)["properties"]

    def strip_non_numeric_chars(s):
        while len(s) > 0 and not s[0].isdigit():
            s = s[1:]
        while len(s) > 0 and not s[-1].isdigit():
            s = s[:-1]
        return s

    async def create_json_object(target: str, schema, get_value: Callable, current_obj=None, root_obj=None, parent_key='', prefix=''):
        if current_obj is None:
            current_obj = {}
            root_obj = {}

        for name, properties in schema.items():
            full_key = f'{parent_key}.{name}' if parent_key else name
            if properties['type'] == 'object':
                current_obj[name] = await create_json_object(target, properties['properties'], get_value, {}, root_obj, full_key)
            else:
                description = properties.get('description')
                enum = properties.get('enum', None)
                if prefix:
                    description = prefix + (description[:1].lower() + description[1:])
                value = await get_value(target, full_key, properties['type'], description, enum, root_obj)
                current_obj[name] = value
                root_obj[full_key] = value

        return current_obj

    async def extract_value(text: str, field_name: str, field_type: str, description: str, enum: list, existing_values: dict) -> Any:
        """
        Extract value of type `field_type` from `text` based on `description`.
        `{}` placeholders in `description` will be formatted using `existing_values` dict which has flat dot notation keys
        (e.g. person.age if there is a person object with an age field).
        """
        if enum:
            # if an enum of values is given for the field, perform a classification instead of extraction
            return (await classifier.classify(text, enum)).max_label

        question = description + "?" if not description.endswith("?") else description
        question = question.format(**existing_values)
        question = question[:1].upper() + question[1:]  # capitalize first word

        qa_result = await qa.answer(text, question, confidence_threshold=0.01)

        if qa_result is None:
            return None

        if field_type == "integer":
            try:
                return int(strip_non_numeric_chars(qa_result.answer))
            except ValueError:
                return None
        elif field_type == "number":
            try:
                return float(strip_non_numeric_chars(qa_result.answer))
            except ValueError:
                return None
        else:
            return qa_result.answer.strip(" .,;:!?")

    if not multiple:
        return await create_json_object(input_md, output_schema, extract_value)
    else:
        if not multiple_description or multiple_description.isspace():
            raise MissingParameterError("Description for entity type is required.")

        result = await token_classifier.classify(input_md, classes=[multiple_description], confidence_threshold=0.75)
        entities = [s.word for s in result.spans]

        # to specify the current entity we are interested in, we mark it in the context and prepend a hint to the description
        description_prefix = f"For the {multiple_description} marked by << >>, "
        extracted = [
            await create_json_object(input_md.replace(entity, f"<< {entity} >>"), output_schema, extract_value, prefix=description_prefix)
            for entity in entities
        ]

        def clean_dict_strings(d):
            for key, value in d.items():
                if isinstance(value, str):
                    d[key] = re.sub(r'<<\s|\s>>', '', value)
            return d

        extracted = [clean_dict_strings(e) for e in extracted]

        # deduplicate
        extracted = [e for i, e in enumerate(extracted) if e not in extracted[:i]]
        # return only objects with at least one field having a value
        return [e for e in extracted if any(e.values())]

