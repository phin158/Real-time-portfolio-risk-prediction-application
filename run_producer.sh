#!/bin/bash
echo "Starting Data Producer (Mock Streaming)..."
source .venv/bin/activate
PYTHONPATH=. python data_pipeline/producer.py
