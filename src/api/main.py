import logging

from fastapi import FastAPI

from .routes import flags, messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ChatSentry API", version="0.1.0")

app.include_router(messages.router)
app.include_router(flags.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
