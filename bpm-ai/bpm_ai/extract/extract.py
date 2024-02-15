import re
import string
from typing import Callable, Any

from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.classification.transformers_classifier import TransformersClassifier, DEFAULT_MODEL_MULTI, \
    DEFAULT_MODEL_EN
from bpm_ai_core.extractive_qa.question_answering import ExtractiveQA
from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.llm.common.message import ToolCallsMessage
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.pos.spacy_pos_tagger import SpacyPOSTagger
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.tracing.decorators import trace
from bpm_ai_core.util.json import expand_simplified_json_schema
from bpm_ai_core.util.language import indentify_language

from bpm_ai.common.json_utils import json_to_md
from bpm_ai.common.multimodal import prepare_audio


@trace("bpm-ai-extract", ["llm"])
async def extract_llm(
    llm: LLM,
    input_data: dict[str, str | dict],
    output_schema: dict[str, str | dict],
    multiple: bool = False,
    multiple_description: str = "",
    asr: ASRModel | None = None
) -> dict | list[dict]:
    def transform_result(**extracted):
        def empty_to_none(v):
            return None if v in ["", "null"] else v

        if multiple and "entities" in extracted.keys():
            extracted = extracted["entities"]

        if isinstance(extracted, list):
            return [transform_result(**d) for d in extracted]
        else:
            return {k: empty_to_none(v) for k, v in extracted.items()}

    tool = Tool.from_callable(
        "information_extraction",
        f"Extracts the relevant {'entities' if multiple else 'information'} from the passage.",
        args_schema={
            "entities": {"type": "array", "description": multiple_description, "items": output_schema}
        } if multiple else output_schema,
        callable=transform_result
    )

    #input_data = prepare_images(input_data)  todo enable once GPT-4V is stable
    input_data = prepare_audio(input_data, asr)

    input_md = json_to_md(input_data).strip()

    prompt = Prompt.from_file("extract", input=input_md)

    result = await llm.predict(prompt, tools=[tool])

    if isinstance(result, ToolCallsMessage):
        return result.tool_calls[0].invoke()
    else:
        return {}


def create_json_object(target: str, schema, get_value: Callable, current_obj=None, root_obj=None, parent_key='', prefix=''):
    if current_obj is None:
        current_obj = {}
        root_obj = {}

    for name, properties in schema.items():
        full_key = f'{parent_key}.{name}' if parent_key else name
        if properties['type'] == 'object':
            current_obj[name] = create_json_object(target, properties['properties'], get_value, {}, root_obj, full_key)
        else:
            description = properties.get('description')
            if prefix:
                description = prefix + (description[:1].lower() + description[1:])
            value = get_value(target, full_key, properties['type'], description, root_obj)
            current_obj[name] = value
            root_obj[full_key] = value

    return current_obj


def filter_and_join(tags, tags_to_join=['NOUN', 'PROPN', 'NUM', 'SYM', 'X']):
    result = []
    current_word = ""
    prev_tag = None

    for token, tag in tags:
        if tag in tags_to_join:
            if prev_tag not in tags_to_join and current_word:
                result.append(current_word.strip())
                current_word = ""
            current_word += token
        else:
            if current_word:
                result.append(current_word.strip())
                current_word = ""
        prev_tag = tag

    if current_word:
        result.append(current_word.strip())
    return result


@trace("bpm-ai-extract", ["extractive-qa"])
async def extract_qa(
    extractive_qa: ExtractiveQA,
    input_data: dict[str, str | dict],
    output_schema: dict[str, str | dict],
    multiple: bool = False,
    multiple_description: str = "",
    asr: ASRModel | None = None
) -> dict | list[dict]:
    #input_data = prepare_images(input_data)  todo enable once GPT-4V is stable
    input_data = prepare_audio(input_data, asr)

    input_md = json_to_md(input_data).strip()
    output_schema = expand_simplified_json_schema(output_schema)["properties"]

    def extract_value(text: str, field_name: str, field_type: str, description: str, existing_values: dict) -> Any:
        """
        Extract value of type `field_type` from `text` based on `description`.
        `{}` placeholders in `description` will be formatted using `existing_values` dict which has flat dot notation keys
        (e.g. person.age if there is a person object with an age field).
        """
        question = description + "?" if not description.endswith("?") else description
        question = question.format(**existing_values)
        question = question[:1].upper() + question[1:]  # capitalize first word

        answer = extractive_qa.answer(text, question, confidence_threshold=0.01)

        if answer is None:
            return None

        if field_type == "integer":
            potential_bad_characters = string.punctuation + string.ascii_letters + string.whitespace
            answer = answer.strip(potential_bad_characters)
            try:
                return int(answer)
            except ValueError:
                return None
        elif field_type == "float":
            potential_bad_characters = string.punctuation + string.ascii_letters + string.whitespace + "€äöüß"  # todo not robust for non-english
            answer = answer.strip(potential_bad_characters)
            try:
                return float(answer)
            except ValueError:
                return None
        else:
            return answer.strip(" .,;:!?")

    if not multiple:
        return create_json_object(input_md, output_schema, extract_value)
    else:
        language = indentify_language(input_md)
        tagger = SpacyPOSTagger(language=language)
        classifier = TransformersClassifier(model=DEFAULT_MODEL_EN if language == "en" else DEFAULT_MODEL_MULTI)

        pos_tags = tagger.tag(input_md)
        candidates = filter_and_join(pos_tags)

        if not multiple_description:
            raise Exception("Description for entity type is required.")

        entities = []
        for candidate in candidates:
            true_label = multiple_description.lower()
            false_label = f"not {true_label}"
            result = classifier.classify(candidate, [true_label, false_label], confidence_threshold=0.75)
            if result == true_label:
                entities.append(candidate)

        # to specify the current entity we are interested in, we mark it in the context and prepend a hint to the description
        description_prefix = f"For the {multiple_description} marked by << >>, "
        extracted = [
            create_json_object(input_md.replace(entity, f"<< {entity} >>"), output_schema, extract_value, prefix=description_prefix)
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

