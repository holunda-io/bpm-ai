import logging

from bpm_ai_core.tracing.tracer import Tracer

try:
    from langfuse import Langfuse
    from langfuse.client import StatefulTraceClient, StatefulSpanClient, StatefulGenerationClient
    has_langfuse = True
except ImportError:
    has_langfuse = False

logger = logging.getLogger(__name__)


class LangfuseTracer(Tracer):
    """
    `Langfuse` Tracer (https://langfuse.com)

    To use, you should have the ``langfuse`` python package installed, and the
    environment variables ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_PUBLIC_KEY`` set.
    """

    def __init__(self):
        if not has_langfuse:
            raise ImportError('langfuse is not installed')
        self.langfuse = Langfuse()

        self.trace: StatefulTraceClient | None = None
        self.implicit_trace: bool = False
        self.span_stack: list[StatefulSpanClient] = []
        self.generation: StatefulGenerationClient | None = None
        self.current_tool = None

    def start_trace(self, name: str, inputs: dict | list[dict], tags: list[str] = None):
        if self.trace:
            raise Exception("Trace already started for this thread - end it first")
        self.trace = self.langfuse.trace(
            name=name,
            input=inputs,
            tags=tags
        )

    def end_trace(self, outputs: dict, error_msg: str = None):
        if not self.trace:
            raise Exception("No trace started for this thread")
        self.trace.update(
            output=outputs,
            level="ERROR" if error_msg else None,
            status_message=error_msg
        )
        logger.info(f"[Langfuse Trace Finished] {self.trace.get_trace_url()}")
        self.trace = None
        self.span_stack = []

    def start_span(self, name: str, inputs: dict = None):
        if not self.trace:
            raise Exception("No trace started for this thread")
        if self.span_stack:
            client = self.span_stack[-1]
        else:
            client = self.trace
        span = client.span(name=name, input=inputs)
        self.span_stack.append(span)

    def end_span(self, output=None, error_msg: str = None):
        if not self.span_stack:
            raise Exception("No span started for this thread")
        span = self.span_stack.pop()
        span.end(
            output=output,
            level="ERROR" if error_msg else None,
            status_message=error_msg
        )

    def start_llm_trace(self, llm, messages: list, current_try: int, tools=None):
        if not self.trace and not self.span_stack:
            self.start_trace("unnamed", inputs=messages)
            self.implicit_trace = True
            client = self.trace
        elif self.span_stack:
            client = self.span_stack[-1]
        else:
            client = self.trace

        self.generation = client.generation(
            model=llm.model,
            model_parameters={
                "temperature": llm.temperature
            },
            input=messages,
            metadata={
                "current_try": current_try,
                "max_tries": llm.max_retries + 1,
                "tools": tools
            }
        )

    def end_llm_trace(self, completion=None, error_msg: str = None):
        if not self.generation:
            raise Exception("No generation started for this thread")
        self.generation.end(
            output=completion,
            level="ERROR" if error_msg else None,
            status_message=error_msg
        )
        if self.implicit_trace:
            self.end_trace(outputs=completion)

    def start_tool_trace(self, tool, inputs: dict):
        self.start_span(tool.name, inputs)

    def end_tool_trace(self, outputs: dict | None = None, error_msg: str | None = None):
        self.end_span(outputs, error_msg)

    def event(self, name: str, inputs: dict | None = None, outputs: dict | None = None, error_msg: str | None = None):
        if not self.trace:
            raise Exception("No trace started for this thread")
        if self.span_stack:
            client = self.span_stack[-1]
        else:
            client = self.trace
        client.event(
            name=name,
            inputs=inputs,
            output=outputs,
            level="ERROR" if error_msg else None,
            status_message=error_msg
        )

    def finalize(self):
        self.langfuse.flush()
