from bpm_ai_core.tracing.tracer import Tracer


class DelegateTracer(Tracer):
    def __init__(self, tracers: list[Tracer]):
        self.tracers = tracers

    def start_trace(self, name, inputs):
        for tracer in self.tracers:
            tracer.start_trace(name, inputs)

    def end_trace(self, outputs, error_msg=None):
        for tracer in self.tracers:
            tracer.end_trace(outputs)

    def start_span(self, name, inputs):
        for tracer in self.tracers:
            tracer.start_span(name, inputs)

    def end_span(self, outputs, error_msg=None):
        for tracer in self.tracers:
            tracer.end_span(outputs, error_msg)

    def start_llm_trace(self, llm, messages, current_try, tools=None):
        for tracer in self.tracers:
            tracer.start_llm_trace(llm, messages, current_try, tools)

    def end_llm_trace(self, completion=None, error_msg=None):
        for tracer in self.tracers:
            tracer.end_llm_trace(completion, error_msg)

    def start_tool_trace(self, tool, inputs):
        for tracer in self.tracers:
            tracer.start_tool_trace(tool, inputs)

    def end_tool_trace(self, output=None, error_msg=None):
        for tracer in self.tracers:
            tracer.end_tool_trace(output, error_msg)

    def event(self, name: str, inputs: dict | None = None, outputs: dict | None = None, error_msg: str | None = None):
        for tracer in self.tracers:
            tracer.event(name, inputs, outputs, error_msg)

    def finalize(self):
        for tracer in self.tracers:
            tracer.finalize()
