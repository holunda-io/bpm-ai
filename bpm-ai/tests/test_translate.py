import pytest
from bpm_ai_core.llm.common.message import AssistantMessage
from bpm_ai_core.testing.fake_llm import FakeLLM

from bpm_ai.common.errors import MissingParameterError, FileNotSupportedError
from bpm_ai.translate.translate import translate_llm, translate_nmt


async def test_translate(llm):
    input_data = {
        "email": "Hey ich bins, der Jürgen. Ich habe ein neues Auto.",
        "subject": "Hallo!",
    }
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(
                content={"email": "Hey it\'s me, Jürgen. I have a new car.", "subject": "Hello!"}
            )
        ]
    )
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="English",
    )

    if isinstance(llm, FakeLLM):
        llm.assert_last_request_contains("Jürgen")

    assert "car" in result["email"]
    assert result["subject"] == "Hello!"


async def test_translate_partial_none(llm):
    input_data = {
        "email": None,
        "subject": "Hallo!"
    }
    llm = llm or FakeLLM(
        name="openai",
        responses=[
            AssistantMessage(
                content={"email": None, "subject": "Hello!"}
            )
        ]
    )
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="English",
    )

    if isinstance(llm, FakeLLM):
        llm.assert_last_request_contains("subject")
        llm.assert_last_request_not_contains("email")

    assert result["email"] is None
    assert result["subject"] == "Hello!"


async def test_translate_none(llm):
    input_data = {
        "email": None,
        "subject": None
    }
    llm = llm or FakeLLM(name="openai")
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="English",
    )

    # LLM should not be used if input is all None
    if isinstance(llm, FakeLLM):
        llm.assert_no_request()

    assert result["email"] is None
    assert result["subject"] is None


async def test_translate_image(llm):
    input_data = {
        "doc": "files/invoice.png",
    }
    llm = llm or FakeLLM(
        name="openai",
        supports_images=True,
        responses=[
            AssistantMessage(
                content={"doc": "Rechnung\n\nVon:\nDEMO - Sliced Invoices\nSuite 5A-1204\n123 Somewhere Street\nYour City AZ 12345\nadmin@slicedinvoices.com\n\nRechnungsnummer: INV-3337"}
            )
        ]
    )
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="German",
    )
    #if isinstance(llm, FakeLLM):
    #    llm.assert_last_request_contains("admin@slicedinvoices.com")

    assert "Rechnung" in result["doc"]


async def test_translate_empty(llm):
    input_data = {}
    llm = llm or FakeLLM(name="openai")
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="English",
    )

    # LLM should not be used if input is empty
    if isinstance(llm, FakeLLM):
        llm.assert_no_request()

    assert result == {}


async def test_translate_no_language(llm):
    input_data = {
        "email": "Hey",
        "subject": "Hallo"
    }
    llm = llm or FakeLLM(name="openai")
    target_language = " "

    with pytest.raises(MissingParameterError):
        await translate_llm(
            llm=llm,
            input_data=input_data,
            target_language=target_language,
        )


async def test_translate_unsupported_file(llm):
    llm = llm or FakeLLM(
        name="openai",
        responses=[]
    )
    with pytest.raises(FileNotSupportedError):
        await translate_llm(
            llm=llm,
            input_data={
                "doc": "files/document.docx"
            },
            target_language="German",
        )
