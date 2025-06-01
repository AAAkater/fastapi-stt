from fastapi import FastAPI

from app.api import api_router

app = FastAPI(title="cuit stt")

app.include_router(api_router)
