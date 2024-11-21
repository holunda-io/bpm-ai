import pytest
from bpm_ai_core.llm.common.message import ChatMessage, AssistantMessage
from bpm_ai_core.llm.openai_chat.openai_chat import ChatOpenAI
from bpm_ai_core.testing.fake_llm import FakeLLM, tool_response

from bpm_ai.common.errors import FileNotSupportedError
from bpm_ai.compose.compose import compose_llm


async def test_compose(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(
                content={
                    "greet_customer": "Hey Max",
                    "thank_customer_for_mail": "Thanks for your mail",
                    "answer_question_based_provided_answer": "Your order was shipped today!"
                }
            )
        ]
    )
    result = await compose_llm(
        llm=llm,
        input_data={
            "email": "Hey, where is my order? Max",
            "answer": "Shipped today",
            "agent_name": "Lisa",
        },
        template="{greet customer}, {thank customer for mail}.\n{answer question based on provided answer}.\nBest,\n{agent_name}",
        properties={
            "language": "English",
            "type": "letter",
            "tone": "friendly",
            "length": "short",
            "style": "formal",
            "temperature": "0"
        }
    )

    if isinstance(llm, FakeLLM):
        llm.assert_last_request_contains("Shipped today")

    assert "Max" in result["text"]
    assert "Lisa" in result["text"]
    assert "shipped" in result["text"]


async def test_compose_empty_template(llm):
    template = ""
    llm = llm or FakeLLM(name="openai")
    result = await compose_llm(
        llm=llm,
        input_data={"email": "Hey, where is my order? Max"},
        template=template,
        properties={
            "language": "English",
            "type": "letter",
            "tone": "friendly",
            "length": "short",
            "style": "formal",
            "temperature": "0"
        }
    )

    # LLM should not be used if template is empty
    if isinstance(llm, FakeLLM):
        llm.assert_no_request()

    assert result["text"] == ""


async def test_compose_unsupported_file(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[]
    )
    with pytest.raises(FileNotSupportedError):
        await compose_llm(
            llm=llm,
            input_data={
                "doc": "files/document.docx"
            },
            template="{test}",
            properties={}
        )