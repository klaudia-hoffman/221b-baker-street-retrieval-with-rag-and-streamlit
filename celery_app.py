import config  # noqa: F401  # loads .env and validates settings

from celery import Celery
from config import settings

celery_app = Celery(
    "rag_worker",
    broker=settings.rabbitmq_url,
    backend="rpc://",
    include=["tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
