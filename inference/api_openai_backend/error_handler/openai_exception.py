from fastapi import Request, status
from starlette.responses import JSONResponse

from ..openai_protocol import ModelResponse
from ..openai_protocol import ErrorResponse
import traceback


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

def openai_exception_handler(request: Request, exc: OpenAIHTTPException):
    assert isinstance(
        exc, OpenAIHTTPException
    ), f"Unable to handle invalid exception {type(exc)}"
    if exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        message = f"Internal Server Error (Request ID: {request.state.request_id})"
        internal_message = message
        exc_type = "InternalServerError"
    else:
        internal_message = extract_message_from_exception(exc)
        message = exc.message
        exc_type = exc.type
    err_response = ModelResponse(
        error=ErrorResponse(
            message=message,
            code=exc.status_code,
            internal_message=internal_message,
            type=exc_type,
        )
    )
    return JSONResponse(content=err_response.dict(), status_code=exc.status_code)

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