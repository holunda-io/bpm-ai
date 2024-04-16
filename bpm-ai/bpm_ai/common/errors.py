
class BpmAiError(Exception):
    pass


class MissingParameterError(BpmAiError):
    pass


class LanguageNotFoundError(BpmAiError):
    pass


class FileNotSupportedError(BpmAiError):
    pass
