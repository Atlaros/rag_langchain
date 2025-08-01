#!/usr/bin/env bash
set -e

# activa virtualenv
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# actualiza e instala deps
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# corre el servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
