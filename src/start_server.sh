#!/bin/bash
"""
Shell script to start the FastAPI backend server
"""

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the server
echo "Starting FastAPI backend server..."
uvicorn src.api.backend:app --host 127.0.0.1 --port 8000 --reload