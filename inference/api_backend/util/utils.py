import warnings

import traceback
def _replace_prefix(model: str) -> str:
    return model.replace("--", "/")



try:
    from langchain.llms import OpenAIChat

    LANGCHAIN_INSTALLED = True
    LANGCHAIN_SUPPORTED_PROVIDERS = {"openai": OpenAIChat}
except ImportError:
    LANGCHAIN_INSTALLED = False

def _get_langchain_model(model: str):
    if not LANGCHAIN_INSTALLED:
        raise ValueError(
            f"Unsupported model {model}. If you want to use a langchain-"
            "compatible model, install langchain ( pip install langchain )."
        )

    provider, model_name = model.split("://", 1)
    if provider not in LANGCHAIN_SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown model provider for {model}. Supported providers are: "
            f"{' '.join(LANGCHAIN_SUPPORTED_PROVIDERS.keys())}"
        )
    return LANGCHAIN_SUPPORTED_PROVIDERS[provider](model_name=model_name)



def _is_aviary_model(model: str) -> bool:
    """
    Determine if this is an aviary model. Aviary
    models do not have a '://' in them.
    """
    return "://" not in model

def has_backend():
    try:
        import aviary.backend  # noqa: F401

        return True
    except ImportError:
        warnings.warn(traceback.format_exc(), stacklevel=2)
        return False


def assert_has_backend():
    # TODO: aviary is not on pypi yet, instruction not actionable
    assert has_backend(), (
        "This command requires aviary backend to be installed. "
        "Please install backend dependencies with `pip install aviary[backend]`. "
    )


def extract_message_from_exception(e: Exception) -> str:
    # If the exception is a Ray exception, we need to dig through the text to get just
    # the exception message without the stack trace
    # This also works for normal exceptions (we will just return everything from
    # format_exception_only in that case)
    message_lines = traceback.format_exception_only(type(e), e)[-1].strip().split("\n")
    message = ""
    # The stack trace lines will be prefixed with spaces, so we need to start from the bottom
    # and stop at the last line before a line with a space
    found_last_line_before_stack_trace = False
    for line in reversed(message_lines):
        if not line.startswith(" "):
            found_last_line_before_stack_trace = True
        if found_last_line_before_stack_trace and line.startswith(" "):
            break
        message = line + "\n" + message
    message = message.strip()
    return message


class OpenAIHTTPException(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        type: str = "Unknown",
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.type = type