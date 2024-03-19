import logging

from bpm_ai_core.tracing.tracer import Tracer

logger = logging.getLogger("logging-tracer")
logger.setLevel(logging.INFO)


class LoggingTracer(Tracer):

    def __init__(self):
        self._level = 0

    def _indent(self):
        return ("|--" if self._level == 0 else "|") + ("  " * self._level) + ("|--" if self._level > 0 else "")

    def start_trace(self, name: str, inputs: dict, tags: list[str] = None):
        logger.info(f"[TRACE START] {name},{f'{tags}, ' if tags else ''} inputs={inputs}")

    def end_trace(self, outputs, error_msg: str = None):
        if error_msg:
            logger.error(f"[TRACE ERROR] {error_msg}")
        else:
            logger.info(f"[TRACE END] outputs={outputs}")

    def start_span(self, name: str, inputs: dict):
        logger.info(self._indent() + f"[SPAN START] {name}, inputs={inputs}")
        self._level += 1

    def end_span(self, outputs: dict, error_msg: str = None):
        self._level -= 1
        if error_msg:
            logger.error(f"[SPAN ERROR] {error_msg}")
        else:
            logger.info(self._indent() + f"[SPAN END] outputs={outputs}")

    def event(self, name: str, inputs: dict = None, outputs: dict = None, error_msg: str = None):
        if error_msg:
            logger.error(self._indent() + f"[EVENT] {error_msg}")
        else:
            logger.info(self._indent() + f"[EVENT] {name}, inputs={inputs}, outputs={outputs}")

    def start_llm_trace(self, llm, messages, current_try, tools=None):
        logger.info(self._indent() + f"[LLM <] {llm.model}, current_try: {current_try}, tools={tools}, messages={messages}")

    def end_llm_trace(self, completion=None, error_msg=None):
        if error_msg:
            logger.error(self._indent() + f"[LLM COMPLETION ERROR] {error_msg}")
            return
        logger.info(self._indent() + f"[LLM >] {completion}")

    def start_tool_trace(self, tool, inputs):
        logger.info(self._indent() + f"[TOOL] {tool.name}({inputs})")

    def end_tool_trace(self, output=None, error_msg=None):
        if error_msg:
            logger.error(self._indent() + f"[TOOL ERROR] {error_msg}")
        else:
            logger.info(self._indent() + f"[TOOL RESULT] {output}")

    def finalize(self):
        pass
