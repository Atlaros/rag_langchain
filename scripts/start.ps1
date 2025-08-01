# habilita ejecución temporal si falta
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# activa el virtualenv
if (Test-Path ".venv/Scripts/Activate.ps1") {
  . .\.venv\Scripts\Activate.ps1
}

# actualiza/install deps si es necesario
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# carga .env automáticamente (si usas python-dotenv en tu app lo leerá)
# corre el servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
