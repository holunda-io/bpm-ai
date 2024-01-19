from abc import ABCMeta, abstractmethod


class Tracer(metaclass=ABCMeta):
    @abstractmethod
    def start_trace(self, name: str, inputs: dict):
        pass

    @abstractmethod
    def end_trace(self, outputs: dict, error_msg: str | None = None):
        pass

    @abstractmethod
    def start_span(self, name: str, inputs: dict):
        pass

    @abstractmethod
    def end_span(self, outputs: dict, error_msg: str | None = None):
        pass

    @abstractmethod
    def start_llm_trace(self, llm, messages, current_try, tools=None):
        pass

    @abstractmethod
    def end_llm_trace(self, completion=None, error_msg=None):
        pass

    @abstractmethod
    def start_tool_trace(self, tool, inputs: dict):
        pass

    @abstractmethod
    def end_tool_trace(self, outputs: dict | None = None, error_msg: str | None = None):
        pass

    @abstractmethod
    def event(self, name: str, inputs: dict | None = None, outputs: dict | None = None, error_msg: str | None = None):
        pass

    @abstractmethod
    def finalize(self):
        pass
