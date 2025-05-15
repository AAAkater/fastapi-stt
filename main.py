from fastapi import FastAPI

from app.api import api_router
from app.utils.model import init_models

app = FastAPI(title="cuit stt", lifespan=init_models)

app.include_router(api_router)
