#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "Benjamin AI Strategy Assistant - Startup Script"
echo "=========================================================="

# 1. Check for Python (3.9+ is fine)
PYTHON_BIN=""
if [ -d ".venv" ]; then
  # Virtual environment already exists, use it
  PYTHON_BIN="./.venv/bin/python"
  echo "✓ Using existing virtual environment"
else
  # Need to create venv - find any Python 3.x
  for py_cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v $py_cmd &> /dev/null; then
      PYTHON_BIN=$py_cmd
      echo "✓ Found Python: $($py_cmd --version)"
      break
    fi
  done
  
  if [ -z "$PYTHON_BIN" ]; then
    echo "=========================================================="
    echo "❌ Error: Python 3 could not be found."
    echo ""
    echo "Benjamin requires Python 3.9 or higher."
    echo ""
    echo "HOW TO FIX THIS (Mac):"
    echo "1. Install Homebrew if you don't have it:"
    echo '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""
    echo "2. Install Python:"
    echo "   brew install python@3.12"
    echo ""
    echo "Then come back and run this script again!"
    echo "=========================================================="
    exit 1
  fi
fi

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

# Clear proxy variables to avoid boto3 connection issues
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
unset NO_PROXY no_proxy SOCKS_PROXY socks_proxy SOCKS5_PROXY socks5_proxy
unset GIT_HTTP_PROXY GIT_HTTPS_PROXY

# Open browser after a short delay (in background)
(sleep 3 && open "http://${HOST}:${PORT}") &

./.venv/bin/uvicorn backend:app --host $HOST --port $PORT --reload
