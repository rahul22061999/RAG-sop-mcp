from fastapi import FastAPI
from api.websocket.router import router as chat_router

app = FastAPI()


app.include_router(chat_router, tags=["Chat router"])


