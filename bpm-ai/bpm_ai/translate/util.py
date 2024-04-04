from enum import Enum


def get_translation_output_schema(input_data: dict, target_language: str):
    return {k: f'{k} translated into {target_language}' for k in input_data.keys()}


class Language(Enum):
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    DANISH = "da"
    SWEDISH = "sv"
    NYNORSK = "no"
    NORWEGIAN = "no"
    BOKMAL = "no"
    FINNISH = "fi"
    POLISH = "pl"
    UKRAINIAN = "uk"


language_dict = {lang.lower(): Language[lang].value for lang in Language.__members__.keys()}


def get_lang_code(language: str) -> str:
    return language_dict[language.strip().lower()]