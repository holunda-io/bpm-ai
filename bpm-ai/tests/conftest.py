import os

import pytest
from bpm_ai_core.llm.anthropic_chat.anthropic_chat import ChatAnthropic
from bpm_ai_core.llm.openai_chat.openai_chat import ChatOpenAI


@pytest.fixture
def local_llm():
    return ChatOpenAI.for_openai_compatible(
        endpoint=os.environ.get("LOCAL_LLM_ENDPOINT"),
        model="local"
    )


@pytest.fixture
def llm():
    return None #ChatAnthropic(model="claude-3-haiku-20240307")
