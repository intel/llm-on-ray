from fastapi import Request, status
from starlette.responses import JSONResponse

from ..common.openai_protocol import ModelResponse
from ..openai_compat.openai_exception import OpenAIHTTPException
from ..util.utils import extract_message_from_exception
from ..common.openai_protocol import ErrorResponse


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
