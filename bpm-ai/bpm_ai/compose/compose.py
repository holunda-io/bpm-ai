import re
from typing import TypedDict, Callable

from bpm_ai_core.llm.common.llm import LLM
from bpm_ai_core.ocr.ocr import OCR
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.speech_recognition.asr import ASRModel
from bpm_ai_core.tracing.decorators import trace

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.common.multimodal import transcribe_audio, prepare_images_for_llm_prompt, ocr_documents
from bpm_ai.compose.util import remove_stop_words, type_to_prompt_type_str

TEMPLATE_VAR_PATTERN = r'\{\s*([^{}\s]+(?:\s*[^{}\s]+)*)\s*\}'


class TextProperties(TypedDict):
    style: str
    type: str
    tone: str
    length: str
    language: str
    temperature: str


@trace("bpm-ai-compose", ["llm"])
async def compose_llm(
    llm: LLM,
    input_data: dict[str, str | dict],
    template: str,
    properties: TextProperties,
    ocr: OCR | None = None,
    asr: ASRModel | None = None
) -> dict:
    if template is None:
        raise MissingParameterError("template is required")

    def desc_to_var_name(desc: str):
        v = remove_stop_words(desc, separator='_')
        return re.sub(r'[^A-Za-z0-9_\'äöüÄÖÜß]+', '', v).lower()

    def format_vars(template: str, f: Callable[[str], str]):
        return re.sub(TEMPLATE_VAR_PATTERN, lambda m: f(m.group(1)), template)

    if not ocr and llm.supports_images():
        input_data = prepare_images_for_llm_prompt(input_data)
    else:
        input_data = await ocr_documents(input_data, ocr)
    input_data = await transcribe_audio(input_data, asr)

    # all variables found in the template
    template_vars = re.findall(TEMPLATE_VAR_PATTERN, template)
    # map of template variables that are not already in the input and need to be generated
    template_vars_to_generate_dict = {desc_to_var_name(desc): desc for desc in template_vars if
                                      desc not in input_data.keys()}
    # input variables that are not present in the template
    non_template_input_var_dict = {k: v for k, v in input_data.items() if k not in template_vars}

    if len(template_vars_to_generate_dict) > 0:
        prompt = Prompt.from_file(
            "compose",
            context=non_template_input_var_dict,
            # remove template braces from variables that are already present in the input to not confuse the model what to generate
            # we do not resolve these variables here to avoid sending that data to the API
            template=format_vars(template, lambda v: v if v in input_data.keys() else '{' + v + '}'),
            type=type_to_prompt_type_str(properties.get("type", "letter")),
            style=properties.get("style", "formal"),
            tone=properties.get("tone", "friendly"),
            length=properties.get("length", "adequate"),
            lang=properties.get("language", "English")
        )

        compose_schema = {
            "name": "store_text",
            "description": "Stores composed text parts for template variables.",
            "type": "object",
            "properties": template_vars_to_generate_dict
        }

        message = await llm.generate_message(prompt, output_schema=compose_schema)

        generated_vars = message.content or {}
    else:
        generated_vars = {}

    input_vars = {desc_to_var_name(k): v for k, v in input_data.items()}
    all_vars = generated_vars | input_vars

    # resolve all template variables using either input or generated values
    result = format_vars(template, lambda v: all_vars[desc_to_var_name(v)])

    return {"text": result}



