import functools
import logging
import time

from config.settings import settings
from fastmcp import FastMCP
from middleware.middleware import UnkeyAuthMiddleware
from tools import all_tools

logger = logging.getLogger(__name__)


def _logged(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        logger.info("Tool called: %s", fn.__name__)
        t0 = time.monotonic()
        try:
            result = await fn(*args, **kwargs)
            logger.info(
                "Tool completed: %s (%.3fs)", fn.__name__, time.monotonic() - t0
            )
            return result
        except Exception as e:
            logger.error(
                "Tool failed: %s (%.3fs) — %s", fn.__name__, time.monotonic() - t0, e
            )
            raise

    return wrapper


def create_mcp_server() -> FastMCP:
    server = FastMCP(
        name=settings.service_name,
    )

    server.add_middleware(
        UnkeyAuthMiddleware(
            root_key=settings.unkey_root_api_key.get_secret_value(),
        )
    )

    for tool in all_tools:
        server.add_tool(_logged(tool))

    return server


mcp = create_mcp_server()
