import os

from bpm_ai_core.tracing.langsmith import LangsmithTracer
from bpm_ai_core.tracing.tracing import NoopTracer

tracer = LangsmithTracer() if os.environ.get("LANGCHAIN_TRACING_V2") else NoopTracer
