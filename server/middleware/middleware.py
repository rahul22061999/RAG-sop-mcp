from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext
from unkey.py.errors.apierror import APIError
from unkey.py.sdk import Unkey


class UnkeyAuthMiddleware(Middleware):
    def __init__(self, root_key: str) -> None:
        self._unkey = Unkey(root_key=root_key)

    async def on_request(
        self,
        context: MiddlewareContext[Any],
        call_next: Any,
    ) -> Any:
        headers = get_http_headers(include={"authorization"}) or {}
        authorization = headers.get("authorization", "")
        scheme, _, api_key = authorization.partition(" ")

        if scheme.lower() != "bearer" or not api_key:
            raise ToolError("Unauthorized: send Authorization: Bearer <api-key>")
        try:
            result = await self._unkey.keys.verify_key_async(key=api_key.strip())
        except APIError as exc:
            raise ToolError(f"Authentication provider error: {exc.message}") from exc

        if not result.data or not result.data.valid:
            raise ToolError(
                f"Unauthorized: invalid or revoked API key (code={result.data.code if result.data else 'unknown'})"
            )

        return await call_next(context)
