#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "Benjamin AI Strategy Assistant - Startup Script"
echo "=========================================================="

# 1. Check for Python 3.12
PYTHON_BIN="python3.12"
if ! command -v $PYTHON_BIN &> /dev/null; then
  # Try basic "python3" and check if it's 3.12
  if command -v python3 &> /dev/null && python3 --version 2>&1 | grep -q 'Python 3.12'; then
    PYTHON_BIN="python3"
  else
    echo "=========================================================="
    echo "❌ Error: Python 3.12 could not be found."
    echo ""
    echo "Benjamin specifically requires Python 3.12."
    echo ""
    echo "HOW TO FIX THIS (Mac):"
    echo "1. If you don't have Homebrew installed, open a new Terminal and paste:"
    echo '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""
    echo "2. Once Homebrew is installed, run:"
    echo "   brew install python@3.12"
    echo ""
    echo "Then come back and run this script again!"
    echo "=========================================================="
    exit 1
  fi
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

# 5. Check LM Studio (Optional but recommended)
LMSTUDIO_URL="${LMSTUDIO_BASE_URL:-http://localhost:1234}"
if curl -s --connect-timeout 2 "${LMSTUDIO_URL}/v1/models" > /dev/null 2>&1; then
  echo "✓ LM Studio is reachable at ${LMSTUDIO_URL}"
else
  echo "ℹ️  LM Studio is not reachable at ${LMSTUDIO_URL}. You will not be able to use local models."
  echo "   (This is fine if you only plan to use AWS Bedrock models)"
fi

# 6. Start the Application
echo "Starting Benjamin..."
echo "=========================================================="
PORT=${PORT:-8000}
HOST=${HOST:-"127.0.0.1"}

./.venv/bin/uvicorn backend:app --host $HOST --port $PORT --reload
