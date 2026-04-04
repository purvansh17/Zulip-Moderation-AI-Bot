import logging

from minio import Minio

from .config import config

logger = logging.getLogger(__name__)


def get_minio_client() -> Minio:
    client = Minio(
        endpoint=config.MINIO_ENDPOINT,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=config.MINIO_SECURE,
    )
    logger.info("MinIO client connected to %s", config.MINIO_ENDPOINT)
    return client
