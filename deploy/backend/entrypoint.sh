#!/bin/sh
set -e

echo "Running database migrations..."
python src/database/database_setup.py setup

echo "Starting backend server..."
exec python start_backend.py
