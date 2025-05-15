from fastapi import APIRouter

from app.api.v1 import stt

v1_router = APIRouter(prefix="/v1")


v1_router.include_router(stt.router)
