import pytest
from bpm_ai_core.llm.openai_chat import ChatOpenAI
from bpm_ai_core.testing.fake_llm import FakeLLM, tool_response
from bpm_ai_core.translation.easy_nmt.easy_nmt import EasyNMT

from bpm_ai.common.errors import MissingParameterError
from bpm_ai.translate.translate import translate_llm, translate_nmt


async def test_translate(use_real_llm=False):
    input_data = {
        "email": "Hey ich bins, der Jürgen. Ich habe ein neues Auto.",
        "subject": "Hallo!"
    }
    llm = FakeLLM(
        name="openai",
        real_llm_delegate=ChatOpenAI() if use_real_llm else None,
        responses=[
            tool_response(
                name="store_translation",
                payload='{"email": "Hey it\'s me, Jürgen. I have a car.", "subject": "Hello!"}'
            )
        ]
    )
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="English",
    )

    llm.assert_last_request_contains("Jürgen")
    llm.assert_last_request_defined_tool("store_translation", is_fixed_tool_choice=True)

    assert "car" in result["email"]
    assert result["subject"] == "Hello!"


async def test_translate_partial_none(use_real_llm=False):
    input_data = {
        "email": None,
        "subject": "Hallo!"
    }
    llm = FakeLLM(
        name="openai",
        real_llm_delegate=ChatOpenAI() if use_real_llm else None,
        responses=[
            tool_response(
                name="store_translation",
                payload='{"email": null, "subject": "Hello!"}'
            )
        ]
    )
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="English",
    )

    llm.assert_last_request_contains("subject")
    llm.assert_last_request_not_contains("email")
    llm.assert_last_request_defined_tool("store_translation", is_fixed_tool_choice=True)

    assert result["email"] is None
    assert result["subject"] == "Hello!"


async def test_translate_none():
    input_data = {
        "email": None,
        "subject": None
    }
    llm = FakeLLM(name="openai")
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="English",
    )

    # LLM should not be used if input is all None
    llm.assert_no_request()

    assert result["email"] is None
    assert result["subject"] is None


async def test_translate_empty():
    input_data = {}
    llm = FakeLLM(name="openai")
    result = await translate_llm(
        llm=llm,
        input_data=input_data,
        target_language="English",
    )

    # LLM should not be used if input is empty
    llm.assert_no_request()

    assert result == {}


async def test_translate_no_language():
    input_data = {
        "email": "Hey",
        "subject": "Hallo"
    }
    llm = FakeLLM(name="openai")
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
