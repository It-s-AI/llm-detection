"""
Some middleware to help with development work, or for extra debugging
"""
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from fiber.logging_utils import get_logger

logger = get_logger(__name__)



async def _logging_middleware(request: Request, call_next) -> Response:
    logger.debug(f"Received request: {request.method} {request.url}")

    try:
        _ = await request.body()
    except Exception as e:
        logger.error(f"Error reading request body: {e}")

    response = await call_next(request)
    if response.status_code != 200:
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        async def new_body_iterator():
            yield response_body

        response.body_iterator = new_body_iterator()
        logger.error(f"Response error content: {response_body.decode()}")
    return response


async def _custom_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"An error occurred: {exc}", exc_info=True)
    return JSONResponse(content={"detail": "Internal Server Error"}, status_code=500)


def configure_extra_logging_middleware(app: FastAPI):
    app.middleware("http")(_logging_middleware)
    app.add_exception_handler(Exception, _custom_exception_handler)
    logger.info("Development middleware and exception handler added.")
