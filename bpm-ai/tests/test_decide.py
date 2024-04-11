from bpm_ai_inference.classification.transformers_classifier import TransformersClassifier
from bpm_ai_core.llm.common.message import AssistantMessage
from bpm_ai_core.testing.fake_llm import FakeLLM

from bpm_ai.decide.decide import decide_llm, decide_classifier


async def test_decide(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(content={"decision": "yup", "reasoning": ""})
        ]
    )
    result = await decide_llm(
        llm=llm,
        input_data={"email": "Hallo ich bins, der John Meier. Mein 30. Geburtstag war gut!"},
        instructions="Is the user older than 18 years?",
        strategy="fast",
        possible_values=["yup", "nope"],
        output_type="string"
    )

    if isinstance(llm, FakeLLM):
        llm.assert_last_request_contains("John Meier")

    assert result["decision"] == "yup"


async def test_decide_image(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(content={"decision": "INVOICE", "reasoning": ""})
        ]
    )
    result = await decide_llm(
        llm=llm,
        input_data={
            "email": "Hey, you can find the document we talked about attached!",
            "doc": "invoice-simple.webp"
        },
        instructions="What kind of document is that?",
        strategy="cot",
        possible_values=["APPLICATION", "COMPLAINT", "INVOICE", "TAXES"],
        output_type="string"
    )

    assert result["decision"] == "INVOICE"


async def test_decide_none(llm):
    input_data = {
        "email": None,
        "subject": None
    }
    llm = llm or FakeLLM(name="openai")
    result = await decide_llm(
        llm=llm,
        input_data=input_data,
        instructions="Is the user older than 18 years?",
        output_type="boolean"
    )

    # LLM should not be used if input is all None
    if isinstance(llm, FakeLLM):
        llm.assert_no_request()

    assert result["decision"] is None
    assert result["reasoning"] == "No input values present."


async def test_decide_classifier():
    classifier = TransformersClassifier()

    result = await decide_classifier(
        classifier=classifier,
        input_data={"email": "Hey its me, John Meier. My 30th birthday was great!"},
        question="What is the person?",
        possible_values=["minor", "adult", "elder", "unclear"],
        output_type="string"
    )

    assert result["decision"] == "adult"


async def test_decide_classifier_boolean():
    classifier = TransformersClassifier()

    result = await decide_classifier(
        classifier=classifier,
        input_data={"email": "Hey its me, John Meier. My 30th birthday was great!"},
        question="Is the person an adult?",
        output_type="boolean"
    )

    assert result["decision"] == True


async def test_decide_classifier_float():
    classifier = TransformersClassifier()

    result = await decide_classifier(
        classifier=classifier,
        input_data={"email": "I got 9.5 points in my exam!"},
        possible_values=[5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
        output_type="number"
    )

    assert result["decision"] == 9.5


async def test_decide_classifier_none():
    input_data = {
        "email": None,
        "subject": None
    }
    classifier = TransformersClassifier()
    result = await decide_classifier(
        classifier=classifier,
        input_data=input_data,
        question="Is the user older than 18 years?",
        output_type="boolean"
    )

    assert result["decision"] is None
    assert result["reasoning"] == "No input values present."
