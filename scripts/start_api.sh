#!/bin/bash
# Start the Aegis API server

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the server
echo "🛡️  Starting Aegis Threat Detection API..."
echo "   API Docs: http://localhost:8000/docs"
echo "   Health: http://localhost:8000/health"
echo ""

uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
