import os
import sqlite3
import hashlib
from typing import Union
from pathlib import Path
from PyPDF2 import PdfReader
from io import BytesIO

# rutas absolutas: la DB está en la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "vector_metadata.db"

def get_connection():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            namespace TEXT,
            source TEXT,
            active INTEGER DEFAULT 1
        )
        """
    )
    return conn

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 50):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - overlap
    return chunks

def _extract_text_from_pdf_obj(pdf_obj) -> str:
    reader = PdfReader(pdf_obj)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

def add_documents_from_pdf_path(pdf_path: str, namespace: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")
    full_text = _extract_text_from_pdf_obj(pdf_path)
    return _index_text_chunks(full_text, namespace, os.path.basename(pdf_path))

def add_documents_from_pdf_bytes(pdf_bytes: Union[bytes, bytearray], namespace: str, source: str = "<in-memory>"):
    pdf_obj = BytesIO(pdf_bytes)
    full_text = _extract_text_from_pdf_obj(pdf_obj)
    return _index_text_chunks(full_text, namespace, source)

def _index_text_chunks(full_text: str, namespace: str, source: str):
    chunks = chunk_text(full_text)
    conn = get_connection()
    cursor = conn.cursor()
    added = 0
    for chunk in chunks:
        chunk_id = hashlib.sha256(f"{namespace}:{chunk}".encode("utf-8")).hexdigest()
        cursor.execute("SELECT 1 FROM chunks WHERE id=? AND active=1", (chunk_id,))
        if cursor.fetchone():
            continue
        cursor.execute(
            "INSERT OR REPLACE INTO chunks (id, content, namespace, source, active) VALUES (?, ?, ?, ?, 1)",
            (chunk_id, chunk, namespace, source),
        )
        added += 1
    conn.commit()
    return {"indexed_chunks": added, "namespace": namespace}

# Compatibilidad de llamada genérica
def add_documents_from_pdf(pdf_input, namespace: str):
    if isinstance(pdf_input, (bytes, bytearray)):
        return add_documents_from_pdf_bytes(pdf_input, namespace, source="<bytes>")
    elif isinstance(pdf_input, str):
        return add_documents_from_pdf_path(pdf_input, namespace)
    else:
        raise ValueError("Unsupported type for add_documents_from_pdf")

def list_namespaces():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT namespace FROM chunks WHERE active=1")
    rows = cursor.fetchall()
    return [r[0] for r in rows if r[0]]
