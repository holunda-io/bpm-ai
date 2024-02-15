from bpm_ai_core.classification.transformers_classifier import TransformersClassifier
from bpm_ai_core.extractive_qa.transformers_qa import TransformersExtractiveQA


def test_qa():
    context = "My name is John and I live in Hawaii"
    question = "Where does John live?"
    expected = "Hawaii"

    qa = TransformersExtractiveQA()
    actual = qa.answer(context, question)

    assert actual.strip() == expected


def test_qa_unanswerable():
    context = "My name is John and I live in Hawaii"
    question = "How much is the fish?"

    qa = TransformersExtractiveQA()
    actual = qa.answer(context, question, confidence_threshold=0.1)

    assert actual is None
