import logging
from typing import Tuple

from bpm_ai_core.pos.pos_tagger import POSTagger

try:
    import spacy
    has_spacy = True
except ImportError:
    has_spacy = False

logger = logging.getLogger(__name__)


def _get_pipeline_for_language(language: str):
    match language:
        case "de":
            return "de_core_news_md"
        case "da":
            return "da_core_news_md"
        case "nl":
            return "nl_core_news_md"
        case "fi":
            return "fi_core_news_md"
        case "fr":
            return "fr_core_news_md"
        case "it":
            return "it_core_news_md"
        case "nn":
            return "nb_core_news_md"
        case "pt":
            return "pt_core_news_md"
        case "es":
            return "es_core_news_md"
        case "sv":
            return "sv_core_news_md"
        case "pl":
            return "pl_core_news_md"
        case "uk":
            return "uk_core_news_md"
        case _:
            return "en_core_web_md"


class SpacyPOSTagger(POSTagger):
    """
    Local POS Tagger based on spaCy library.

    To use, you should have the ``spacy`` python package installed.
    """

    def __init__(self, language: str = "en"):
        if not has_spacy:
            raise ImportError('spacy is not installed')
        pipeline = _get_pipeline_for_language(language)
        for i in range(3):
            try:
                self.nlp = spacy.load(pipeline, disable=["lemmatizer"])
            except OSError:
                from spacy.cli import download
                download(pipeline)
                continue

    def tag(self, text: str) -> list[Tuple[str, str]]:
        doc = self.nlp(text)
        return [(token.text_with_ws, token.pos_) for token in doc]



