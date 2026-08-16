"""
Tests for UnkeyAuthMiddleware's request-rejection paths. This is the only
thing standing between the SOP knowledge base and the open internet on a
publicly reachable MCP endpoint, so its failure modes need real coverage:
missing header, wrong scheme, revoked key, and the provider-error path all
have to fail closed (raise), never silently let a request through.
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp.exceptions import ToolError
from middleware.middleware import UnkeyAuthMiddleware
from unkey.py.errors.apierror import APIError


def _make_middleware(mocker, verify_result=None, verify_side_effect=None):
    mock_unkey_cls = mocker.patch("middleware.middleware.Unkey")
    mock_unkey_cls.return_value.keys.verify_key_async = AsyncMock(
        return_value=verify_result, side_effect=verify_side_effect
    )
    return UnkeyAuthMiddleware(root_key="root-test-key")


def _headers(mocker, value: dict):
    mocker.patch("middleware.middleware.get_http_headers", return_value=value)


class _Result:
    def __init__(self, valid: bool, code: str = "OK"):
        self.data = _Data(valid, code)


class _Data:
    def __init__(self, valid: bool, code: str):
        self.valid = valid
        self.code = code


@pytest.mark.asyncio
async def test_missing_authorization_header_is_rejected(mocker):
    middleware = _make_middleware(mocker)
    _headers(mocker, {})

    with pytest.raises(ToolError, match="Unauthorized"):
        await middleware.on_request(context=mocker.Mock(), call_next=AsyncMock())


@pytest.mark.asyncio
async def test_non_bearer_scheme_is_rejected(mocker):
    middleware = _make_middleware(mocker)
    _headers(mocker, {"authorization": "Basic dXNlcjpwYXNz"})

    with pytest.raises(ToolError, match="Unauthorized"):
        await middleware.on_request(context=mocker.Mock(), call_next=AsyncMock())


@pytest.mark.asyncio
async def test_bearer_with_no_key_is_rejected(mocker):
    middleware = _make_middleware(mocker)
    _headers(mocker, {"authorization": "Bearer "})

    with pytest.raises(ToolError, match="Unauthorized"):
        await middleware.on_request(context=mocker.Mock(), call_next=AsyncMock())


@pytest.mark.asyncio
async def test_revoked_key_is_rejected(mocker):
    middleware = _make_middleware(
        mocker, verify_result=_Result(valid=False, code="REVOKED")
    )
    _headers(mocker, {"authorization": "Bearer sk_revoked"})

    with pytest.raises(ToolError, match="invalid or revoked"):
        await middleware.on_request(context=mocker.Mock(), call_next=AsyncMock())


@pytest.mark.asyncio
async def test_provider_error_is_surfaced_not_swallowed(mocker):
    raw_response = httpx.Response(
        status_code=500,
        json={"error": "internal"},
        request=httpx.Request("POST", "https://api.unkey.dev/v2/keys.verifyKey"),
    )
    api_error = APIError(message="Unkey is down", raw_response=raw_response)
    middleware = _make_middleware(mocker, verify_side_effect=api_error)
    _headers(mocker, {"authorization": "Bearer sk_whatever"})

    with pytest.raises(ToolError, match="Authentication provider error"):
        await middleware.on_request(context=mocker.Mock(), call_next=AsyncMock())


@pytest.mark.asyncio
async def test_valid_key_calls_next_and_returns_its_result(mocker):
    middleware = _make_middleware(mocker, verify_result=_Result(valid=True))
    _headers(mocker, {"authorization": "Bearer sk_valid"})

    call_next = AsyncMock(return_value="tool result")
    context = mocker.Mock()

    result = await middleware.on_request(context=context, call_next=call_next)

    assert result == "tool result"
    call_next.assert_awaited_once_with(context)
