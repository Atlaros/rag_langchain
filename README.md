# RAG Backend con LangChain + HuggingFace

## Características
- Ingesta de PDFs.  
- Almacenamiento de chunks en SQLite con deduplicación por hash.  
- Recuperación y QA usando LangChain: embeddings multilingües (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) y generación con modelo instruccional de Hugging Face (`Mistral-7B-Instruct-v0.3`).  
- Microservicio en FastAPI.

## Setup rápido

1. Clona o descomprime el proyecto y entra al directorio.
2. Crea y activa un virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # en Windows PowerShell: .\.venv\Scripts\Activate.ps1
   ```
3. Instala dependencias:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Copia `.env.template` a `.env` y rellena tu token de HF:
   ```bash
   cp .env.template .env
   ```
5. Arranca el servidor:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Endpoints principales

- `POST /ingest_pdf`: subir y procesar PDF. multipart/form-data: `file`, `namespace`.
- `POST /rag`: hacer consulta RAG. JSON:
  ```json
  {
    "type":"query",
    "user_id":"compromidos",
    "role":"analyst",
    "email_text":"Resume los puntos clave."
  }
  ```

## Variables importantes en .env
- `HUGGINGFACEHUB_API_TOKEN`: tu token de Hugging Face.  
- `EMBEDDING_MODEL`: modelo de embeddings.  
- `GENERATION_MODEL`: modelo de generación.  
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: control de chunking.  
- `RETRIEVAL_K`: cuántos documentos recuperar.  
- `TEMPERATURE`, `MAX_TOKENS`: para generación.

