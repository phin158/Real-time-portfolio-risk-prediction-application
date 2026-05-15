#!/bin/bash
echo "Starting Streamlit Dashboard..."
source .venv/bin/activate
PYTHONPATH=. streamlit run dashboard/app.py
