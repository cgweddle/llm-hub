#!/bin/sh
set -e

echo "Running database migrations..."
python src/database/database_setup.py setup

echo "Seeding default tools..."
python src/tools/seed_tools.py

echo "Starting backend server..."
exec python start_backend.py
