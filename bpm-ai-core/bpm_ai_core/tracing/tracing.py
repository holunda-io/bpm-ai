import os
from contextvars import ContextVar

from bpm_ai_core.tracing.delegate import DelegateTracer

from bpm_ai_core.tracing.tracer import Tracer

_tracers: ContextVar[list[Tracer] | None] = ContextVar('tracers', default=None)


def _configure_tracers():
    tracers = []
    if not os.environ.get("BPM_AI_CONSOLE_TRACING_DISABLED", False):
        from bpm_ai_core.tracing.logging import LoggingTracer
        tracers.append(LoggingTracer())
    if os.environ.get("LANGFUSE_SECRET_KEY"):
        from bpm_ai_core.tracing.langfuse import LangfuseTracer
        tracers.append(LangfuseTracer())
    _tracers.set(tracers)


class Tracing:
    @staticmethod
    def tracers() -> Tracer:
        if _tracers.get() is None:
            _configure_tracers()
        return DelegateTracer(_tracers.get())

    @staticmethod
    def finalize():
        Tracing.tracers().finalize()
