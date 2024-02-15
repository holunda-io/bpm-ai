from bpm_ai_core.llm.openai_chat import ChatOpenAI
from bpm_ai_core.testing.fake_llm import FakeLLM, tool_response
from bpm_ai_core.translation.easy_nmt.easy_nmt import EasyNMT

from bpm_ai.translate.translate import translate_llm, translate_nmt


async def test_translate(use_real_llm=False):
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
        input_data={
            "email": "Hey ich bins, der Jürgen. Ich habe ein neues Auto.",
            "subject": "Hallo!"
        },
        target_language="English",
    )

    llm.assert_last_request_contains("Jürgen")
    llm.assert_last_request_defined_tool("store_translation", is_fixed_tool_choice=True)

    assert "car" in result["email"]
    assert result["subject"] == "Hello!"


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
