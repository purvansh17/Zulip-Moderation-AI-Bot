import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "postgresql://user:chatsentry_pg@localhost:5432/chatsentry",
    )
    MINIO_ENDPOINT: str = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.environ.get("MINIO_ACCESS_KEY", "admin")
    MINIO_SECRET_KEY: str = os.environ.get("MINIO_SECRET_KEY", "chatsentry_minio")
    MINIO_SECURE: bool = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
    BUCKET_RAW: str = "zulip-raw-messages"
    BUCKET_TRAINING: str = "zulip-training-data"


config = Config()
