from abc import ABCMeta, abstractmethod


class Tracer(metaclass=ABCMeta):
    @abstractmethod
    def start_llm_trace(self, llm, messages, current_try, tools):
        pass

    @abstractmethod
    def end_llm_trace(self, completion, error_msg):
        pass

    @abstractmethod
    def start_function_trace(self, function, inputs):
        pass

    @abstractmethod
    def end_function_trace(self, output, error_msg):
        pass

    @abstractmethod
    def start_trace(self, name, run_type, inputs, executor, metadata, tags, client, extra):
        pass

    @abstractmethod
    def end_trace(self, outputs, error):
        pass


class NoopTracer(Tracer):
    def start_llm_trace(self, llm, messages, current_try, tools):
        pass

    def end_llm_trace(self, completion, error_msg):
        pass

    def start_function_trace(self, function, inputs):
        pass

    def end_function_trace(self, output, error_msg):
        pass

    def start_trace(self, name, run_type, inputs, executor, metadata, tags, client, extra):
        pass

    def end_trace(self, outputs, error):
        pass