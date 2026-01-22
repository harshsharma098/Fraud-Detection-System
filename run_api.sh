#!/bin/bash

# Script to run the Fraud Detection API

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API
python src/api.py

