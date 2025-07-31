#!/bin/bash

# Backend startup script

set -e

echo "🚀 Starting Llama4 Backend..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Ollama is running
echo "🤖 Checking Ollama status..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "❌ Ollama is not running. Please start it first:"
    echo "   sudo systemctl start ollama"
    exit 1
fi

# Check if Llama4 model is available
echo "🔍 Checking Llama4 model..."
if ! ollama list | grep -q "llama4"; then
    echo "❌ Llama4 model not found. Please pull it first:"
    echo "   ollama pull llama4"
    exit 1
fi

echo "✅ Starting FastAPI server..."
echo "🌐 Backend will be available at: http://localhost:8001"
echo "🛑 Press Ctrl+C to stop the server"

python run_dev.py