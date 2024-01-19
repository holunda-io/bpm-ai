import threading

from bpm_ai_core.tracing.delegate import DelegateTracer
from bpm_ai_core.tracing.logging import LoggingTracer
from bpm_ai_core.tracing.tracer import Tracer

_local = threading.local()
_local.tracers: list[Tracer] = [LoggingTracer()]


class Tracing:
    @staticmethod
    def add_tracer(tracer: Tracer):
        if not isinstance(tracer, Tracer):
            raise ValueError("tracer must be an instance of Tracer")
        _local.tracers.append(tracer)

    @staticmethod
    def tracers() -> Tracer:
        return DelegateTracer(_local.tracers)

    @staticmethod
    def finalize():
        Tracing.tracers().finalize()
