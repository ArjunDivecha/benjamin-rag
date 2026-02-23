#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "Benjamin AI Strategy Assistant - Startup Script"
echo "=========================================================="

# 1. Check for Python 3.12
PYTHON_BIN=${PYTHON_BIN:-"python3.12"}

if ! command -v $PYTHON_BIN &> /dev/null; then
  echo "Error: $PYTHON_BIN could not be found."
  echo "Benjamin requires Python 3.12. Please install it to continue."
  exit 1
fi

echo "✓ Found Python: $($PYTHON_BIN --version)"

# 2. Setup Virtual Environment
if [ ! -d ".venv" ]; then
  echo "Creating new virtual environment (.venv)..."
  $PYTHON_BIN -m venv .venv
else
  echo "✓ Virtual environment already exists."
fi

# 3. Install Dependencies
echo "Ensuring dependencies are installed..."
./.venv/bin/pip install -q -r requirements.txt
echo "✓ Dependencies ready."

# 4. Check API Keys (.env)
if [ ! -f ".env" ]; then
  echo "----------------------------------------------------------"
  echo "⚠️  No .env file found!"
  echo "Creating one from .env.example..."
  cp .env.example .env
  echo "----------------------------------------------------------"
  echo "ACTION REQUIRED:"
  echo "Please open the '.env' file in this folder and add your"
  echo "BEDROCK_API_KEY before running the application again."
  echo "----------------------------------------------------------"
  exit 1
else
  echo "✓ .env config file found."
fi

# 5. Check Ollama (Optional but recommended)
if ! command -v ollama &> /dev/null; then
  echo "ℹ️  Ollama is not installed. You will not be able to use local models."
  echo "   (This is fine if you only plan to use AWS Bedrock models)"
fi

# 6. Start the Application
echo "Starting Benjamin..."
echo "=========================================================="
PORT=${PORT:-8000}
HOST=${HOST:-"127.0.0.1"}

./.venv/bin/uvicorn backend:app --host $HOST --port $PORT --reload
