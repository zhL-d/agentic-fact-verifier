"""Celery configuration for durable verification workers."""

import os

from celery import Celery

celery_app = Celery(
    "agentic_fact_verifier",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    include=["agentic_fact_verifier.worker_tasks"],
)

celery_app.conf.update(
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    enable_utc=True,
    result_backend=None,
    task_acks_late=True,
    task_default_queue="verification",
    task_ignore_result=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    task_soft_time_limit=int(os.environ.get("RUN_SOFT_TIME_LIMIT_SECONDS", "1500")),
    task_time_limit=int(os.environ.get("RUN_TIME_LIMIT_SECONDS", "1800")),
    timezone="UTC",
    worker_prefetch_multiplier=1,
)
