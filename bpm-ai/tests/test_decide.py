import pytest
from bpm_ai_core.llm.common.message import AssistantMessage
from bpm_ai_core.testing.fake_llm import FakeLLM
from bpm_ai_inference.classification.transformers_text_classifier import TransformersClassifier
from bpm_ai_inference.image_classification.transformers_image_classifier import TransformersImageClassifier

from bpm_ai.common.errors import FileNotSupportedError
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


async def test_decide_multiple(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(content={"decision": ["Support", "Samsung"], "reasoning": ""})
        ]
    )
    result = await decide_llm(
        llm=llm,
        input_data={"email": "Hello, my S8 does not turn on any more. What can I do? Thank you!"},
        instructions="What is the email about?",
        strategy="fast",
        possible_values=["Return Shipments", "Support", "Samsung", "LG", "Apple", "Complaint"],
        multiple_decision_values=True,
        output_type="string"
    )

    if isinstance(llm, FakeLLM):
        llm.assert_last_request_contains("Apple")

    assert set(result["decision"]) == {"Support", "Samsung"}


async def test_decide_image(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(content={"decision": "INVOICE", "reasoning": ""})
        ],
        supports_images=True
    )
    result = await decide_llm(
        llm=llm,
        input_data={
            "email": "Hey, you can find the document we talked about attached!",
            "doc": "files/invoice-simple.webp"
        },
        instructions="What kind of document is that?",
        strategy="cot",
        possible_values=["APPLICATION", "COMPLAINT", "INVOICE", "TAXES"],
        output_type="string"
    )

    assert result["decision"] == "INVOICE"


async def test_decide_text(llm):
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
            "doc": "files/document.txt"
        },
        instructions="What kind of document is that?",
        strategy="cot",
        possible_values=["APPLICATION", "COMPLAINT", "INVOICE", "TAXES"],
        output_type="string"
    )

    #if isinstance(llm, FakeLLM):
    #    llm.assert_last_request_contains("Payment is due within 30 days")

    assert result["decision"] == "INVOICE"


async def test_decide_unsupported_file(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[]
    )
    with pytest.raises(FileNotSupportedError):
        await decide_llm(
            llm=llm,
            input_data={
                "doc": "files/document.docx"
            },
            instructions="What kind of document is that?",
            strategy="cot",
            possible_values=["APPLICATION", "COMPLAINT", "INVOICE", "TAXES"],
            output_type="string"
        )


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


async def test_decide_image_classifier_zero_shot():
    image_classifier = TransformersImageClassifier()

    result = await decide_classifier(
        classifier=None,
        image_classifier=image_classifier,
        input_data={"image": "files/example-text.png"},
        question="What kind of image is that?",
        possible_values=["dummy", "invoice", "dog", "house"],
        output_type="string"
    )

    assert result["decision"] == "dummy"


@pytest.mark.skip
async def test_decide_image_classifier():
    image_classifier = TransformersImageClassifier(model="Benjoyo/test-image-classifier-2", zero_shot=False)

    result = await decide_classifier(
        classifier=None,
        image_classifier=image_classifier,
        input_data={"image": "files/invoice.png"},
        output_type="string"
    )

    assert result["decision"] == "Anderes Dokument"


async def test_decide_classifier_multiple(llm):
    classifier = TransformersClassifier()

    result = await decide_classifier(
        classifier=classifier,
        input_data={"email": "Hello, my S8 does not turn on any more. What can I do? Thank you!"},
        question="What is the email about?",
        possible_values=["Return Shipments", "Support", "Samsung", "LG", "Apple"],
        multiple_decision_values=True,
        output_type="string"
    )

    assert set(result["decision"]) == {"Support", "Samsung"}


async def test_decide_classifier_boolean():
    classifier = TransformersClassifier()

    result = await decide_classifier(
        classifier=classifier,
        input_data={"email": "Hey its me, John Meier. My 30th birthday was great!"},
        question="Is the person an adult?",
        output_type="boolean"
    )

    assert result["decision"] is True


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


async def test_decide_classifier_textfile():
    classifier = TransformersClassifier()

    result = await decide_classifier(
        classifier=classifier,
        input_data={"doc": "files/document.txt"},
        possible_values=["invoice", "letter"],
        output_type="string"
    )

    assert result["decision"] is "invoice"
