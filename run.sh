#!/bin/bash
# Cat Inspect AI — Quick Start
# Run from the cat-inspect-ai/ directory

set -e

echo "========================================="
echo "  Cat Inspect AI — Starting Up"
echo "========================================="

# Check GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set!"
    echo "   Run: export GEMINI_API_KEY='your-key'"
    exit 1
fi

# Check Docker container
if ! docker ps | grep -q vectoraidb; then
    echo "⚠️  VectorAI DB not running. Starting..."
    docker compose up -d
    sleep 3
fi

# Check if venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating venv..."
    source .venv/bin/activate
fi

echo ""
echo "Model: ${GEMINI_MODEL:-gemini-3-flash-preview}"
echo "VectorDB: localhost:50051"
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "========================================="

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
