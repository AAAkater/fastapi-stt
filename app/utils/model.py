from contextlib import asynccontextmanager

from fastapi import FastAPI
from faster_whisper import WhisperModel

from app.core.config import settings
from app.utils.logger import logger

whisper_model: WhisperModel


@asynccontextmanager
async def init_models(_: FastAPI):
    try:
        global whisper_model
        whisper_model = WhisperModel(
            settings.model_path,
            device=settings.device,
        )

    except Exception as e:
        logger.error(f"Fail to load model:{e}")
        exit(0)
    yield
