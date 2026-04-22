from contextlib import asynccontextmanager

from fastapi import FastAPI

import config  # noqa: F401  # loads .env and validates OPENAI_API_KEY
from app.routes import router
from data_processing import load_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_store = load_vector_store()
    yield


app = FastAPI(title="221B Baker Street Retrieval", lifespan=lifespan)
app.include_router(router)
