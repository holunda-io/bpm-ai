import os

import pytest
from bpm_ai_core.llm.anthropic_chat.anthropic_chat import ChatAnthropic
from bpm_ai_core.llm.openai_chat.openai_chat import ChatOpenAI
from bpm_ai_core.util.rpc import remote_object


@pytest.fixture
def local_llm():
    return ChatOpenAI.for_openai_compatible(
        endpoint=os.environ.get("LOCAL_LLM_ENDPOINT"),
        model="local"
    )


@pytest.fixture
def llm():
    return None #remote_object("ChatLlamaCpp", "0.0.0.0", 6666, model="QuantFactory/Phi-3-mini-4k-instruct-GGUF")
