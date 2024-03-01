from bpm_ai_core.classification.transformers_classifier import TransformersClassifier
from bpm_ai_core.llm.openai_chat import ChatOpenAI
from bpm_ai_core.testing.fake_llm import FakeLLM, tool_response

from bpm_ai.decide.decide import decide_llm, decide_classifier


async def test_decide(use_real_llm=False):
    llm = FakeLLM(
        name="openai",
        real_llm_delegate=ChatOpenAI() if use_real_llm else None,
        responses=[
            tool_response(
                name="store_decision",
                payload='{"decision": "yup", "reasoning": null}'
            )
        ]
    )
    result = await decide_llm(
        llm=llm,
        input_data={"email": "Hey ich bins, der John Meier. Mein 30. Geburtstag war gut!"},
        instructions="Is the user older than 18 years?",
        strategy="cot",
        possible_values=["yup", "nope"],
        output_type="string"
    )

    llm.assert_last_request_contains("John Meier")
    llm.assert_last_request_defined_tool("store_decision", is_fixed_tool_choice=True)

    assert result["decision"] == "yup"


async def test_decide_none():
    input_data = {
        "email": None,
        "subject": None
    }
    llm = FakeLLM(name="openai")
    result = await decide_llm(
        llm=llm,
        input_data=input_data,
        instructions="Is the user older than 18 years?",
        output_type="boolean"
    )

    # LLM should not be used if input is all None
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
