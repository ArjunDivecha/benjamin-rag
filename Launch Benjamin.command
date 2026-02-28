#!/bin/bash

# Launch Benjamin RAG - Double-click launcher
# This script starts LM Studio and Benjamin, then opens the browser

cd "$(dirname "$0")"

echo "=========================================================="
echo "🚀 Benjamin RAG Launcher"
echo "=========================================================="

# 1. Start LM Studio server if not running
echo "Checking LM Studio..."
if ! lsof -ti:1234 > /dev/null 2>&1; then
    echo "Starting LM Studio server..."
    $HOME/.lmstudio/bin/lms server start
    echo "✓ LM Studio server started"
    
    # Load Mistral model
    echo "Loading Mistral model..."
    $HOME/.lmstudio/bin/lms load mistral-7b-instruct-v0.2 --yes 2>/dev/null || true
    echo "✓ Model loaded"
else
    echo "✓ LM Studio already running"
fi

# 2. Check if Benjamin backend is already running
if lsof -ti:8000 > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Benjamin is already running on port 8000"
    echo ""
    read -p "Do you want to restart it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping existing instance..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        echo "Opening browser to existing instance..."
        open "http://127.0.0.1:8000"
        exit 0
    fi
fi

# 3. Start Benjamin backend
echo ""
echo "Starting Benjamin backend..."
./run_benjamin.sh

# Keep terminal open
echo ""
echo "=========================================================="
echo "Press Ctrl+C to stop Benjamin"
echo "=========================================================="
