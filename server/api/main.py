from fastapi import FastAPI
from prometheus_client import make_asgi_app
from telemetry import setup_telemetry

setup_telemetry("wms-sop-chat-api")

from api.websocket.router import router as chat_router  # noqa: E402

app = FastAPI()

app.include_router(chat_router, tags=["Chat router"])
app.mount("/metrics", make_asgi_app())
