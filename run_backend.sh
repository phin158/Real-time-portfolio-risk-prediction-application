#!/bin/bash
echo "Starting Backend API (FastAPI)..."
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
