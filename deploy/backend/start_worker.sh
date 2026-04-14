#!/bin/sh
set -e

echo "Running database migrations..."
python src/database/database_setup.py setup

echo "Starting Celery worker..."
exec celery -A src.celery_app:celery_app worker --loglevel=info --concurrency=2
