import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from tools.sop_query_tool import stream_sop_answer

logger = logging.getLogger(__name__)

router = APIRouter()


async def _send_answer(websocket: WebSocket, query: str, top_records: int) -> None:
    async for chunk in stream_sop_answer(query, top_records=top_records):
        await websocket.send_json({"type": "token", "content": chunk})
    await websocket.send_json({"type": "done"})


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    """Return as soon as the client closes the socket.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


@router.websocket("/ws/stream")
async def chat(websocket: WebSocket):
    """Stream an SOP answer for each {"query": "..."} message.

    Server messages: {"type": "token", "content": str} per chunk, then
    {"type": "done"}; or {"type": "error", "content": str}.
    If the client disconnects mid-answer, retrieval/generation is cancelled.
    """
    await websocket.accept()

    try:
        while True:
            try:
                payload = await websocket.receive_json()
                query = str(payload["query"]).strip()
                top_records = int(payload.get("top_records", 5))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                await websocket.send_json(
                    {"type": "error", "content": 'Send JSON like {"query": "..."}'}
                )
                continue

            answer_task = asyncio.create_task(
                _send_answer(websocket, query, top_records)
            )
            disconnect_task = asyncio.create_task(_wait_for_disconnect(websocket))
            try:
                await asyncio.wait(
                    {answer_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in (answer_task, disconnect_task):
                    task.cancel()
                await asyncio.gather(answer_task, disconnect_task, return_exceptions=True)

            if disconnect_task.done() and not disconnect_task.cancelled():
                logger.info("Client disconnected; cancelled answer for %r", query)
                return

            try:
                answer_task.result()
            except asyncio.CancelledError:
                return
            except WebSocketDisconnect:
                return
            except Exception:
                logger.exception("Streaming failed for query %r", query)
                await websocket.send_json({"type": "error", "content": "Search failed"})
    except WebSocketDisconnect:
        pass
