import pytest
from bpm_ai_core.llm.common.message import AssistantMessage
from bpm_ai_core.testing.fake_llm import FakeLLM
from bpm_ai_inference.translation.easy_nmt.easy_nmt import EasyNMT

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.translate.translate import translate_llm, translate_nmt


async def test_translate(llm):
    input_data = {
        "email": "Hey ich bins, der Jürgen. Ich habe ein neues Auto.",
        "subject": "Hallo!"
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


async def test_translate_nmt():
    nmt = EasyNMT()
    result = await translate_nmt(
        nmt=nmt,
        input_data={
            "email": "Hey ich bins, der Jürgen. Ich habe ein neues Auto.",
            "subject": "Hallo, mein Freund!"
        },
        target_language="English",
    )

    assert "car" in result["email"]
    assert result["subject"] == "Hello, my friend!"


async def test_translate_nmt_partial_none():
    input_data = {
        "email": None,
        "subject": "Hallo, mein Freund!"
    }
    nmt = EasyNMT()
    result = await translate_nmt(
        nmt=nmt,
        input_data=input_data,
        target_language="English",
    )

    assert result["email"] is None
    assert result["subject"] == "Hello, my friend!"


async def test_translate_nmt_none():
    input_data = {
        "email": None,
        "subject": None
    }
    nmt = EasyNMT()
    result = await translate_nmt(
        nmt=nmt,
        input_data=input_data,
        target_language="English",
    )

    assert result["email"] is None
    assert result["subject"] is None


async def test_translate_nmt_empty():
    input_data = {}
    nmt = EasyNMT()
    result = await translate_nmt(
        nmt=nmt,
        input_data=input_data,
        target_language="English",
    )

    assert result == {}


async def test_translate_nmt_no_language():
    input_data = {
        "email": "Hey",
        "subject": "Hallo"
    }
    nmt = EasyNMT()
    target_language = " "

    with pytest.raises(MissingParameterError):
        await translate_nmt(
            nmt=nmt,
            input_data=input_data,
            target_language=target_language,
        )
