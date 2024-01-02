from abc import ABCMeta, abstractmethod


class Tracer(metaclass=ABCMeta):
    @abstractmethod
    def start_llm_trace(self, llm, messages, current_try, tools=None):
        pass

    @abstractmethod
    def end_llm_trace(self, completion=None, error_msg=None):
        pass

    @abstractmethod
    def start_function_trace(self, function, inputs):
        pass

    @abstractmethod
    def end_function_trace(self, output=None, error_msg=None):
        pass


class NoopTracer(Tracer):
    def start_llm_trace(self, llm, messages, current_try, tools=None):
        pass

    def end_llm_trace(self, completion=None, error_msg=None):
        pass

    def start_function_trace(self, function, inputs):
        pass

    def end_function_trace(self, output=None, error_msg=None):
        pass
