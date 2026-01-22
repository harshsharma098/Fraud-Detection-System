#!/bin/bash

# Setup script for Fraud Detection System

echo "Setting up Fraud Detection System..."
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models logs

# Generate dataset
echo "Generating transaction dataset..."
python src/data_generator.py

# Train models
echo "Training models..."
python src/train.py

echo ""
echo "Setup complete!"
echo ""
echo "To start the API server, run:"
echo "  source venv/bin/activate"
echo "  python src/api.py"
echo ""
echo "Or use uvicorn:"
echo "  uvicorn src.api:app --host 0.0.0.0 --port 8000"

