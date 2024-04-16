import pytest
from bpm_ai_core.llm.common.message import AssistantMessage
from bpm_ai_core.testing.fake_llm import FakeLLM

from bpm_ai.common.errors import FileNotSupportedError
from bpm_ai.generic.generic import generic_llm


async def test_generic(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(
                content={"firstname": "JOHN", "lastname": "MEIER"}
            )
        ]
    )
    result = await generic_llm(
        llm=llm,
        input_data={
            "email": "Hey ich bins, der John Meier.",
            "doc": "files/document.txt"
        },
        instructions="Extract the information and make it all caps.",
        output_schema={
            "firstname": "the firstname",
            "lastname": "the lastname"
        }
    )

    if isinstance(llm, FakeLLM):
        llm.assert_last_request_contains("John Meier")

    assert result["firstname"] == "JOHN"
    assert result["lastname"] == "MEIER"


async def test_generic_unsupported_file(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[]
    )
    with pytest.raises(FileNotSupportedError):
        await generic_llm(
            llm=llm,
            input_data={
                "doc": "files/document.docx"
            },
            instructions="What kind of document is that?",
            output_schema={"res": "result"}
        )