#!/usr/bin/env python3
"""
Startup script for the FastAPI backend server
"""
import uvicorn
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    # Start the uvicorn server
    uvicorn.run(
        "src.api.backend:app",
        host=os.getenv("UVICORN_HOST", "127.0.0.1"),
        port=int(os.getenv("UVICORN_PORT", "8000")),
        reload=os.getenv("UVICORN_RELOAD", "true").lower() == "true",
        log_level="info",
        workers=int(os.getenv("UVICORN_WORKERS", "1")),
    )