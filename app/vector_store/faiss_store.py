import os
import sqlite3
import hashlib
import logging
from typing import Union, List, Dict, Optional
from pathlib import Path
from PyPDF2 import PdfReader
from io import BytesIO

# Configurar logging
logger = logging.getLogger(__name__)

# rutas absolutas: la DB está en la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "vector_metadata.db"

def get_connection():
    """Obtiene conexión a la base de datos SQLite con las tablas necesarias"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    
    # Crear tabla de chunks si no existe
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            namespace TEXT,
            source TEXT,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    
    # Crear tabla de documentos para metadatos adicionales
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            namespace TEXT NOT NULL,
            file_size INTEGER,
            content_hash TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            status TEXT DEFAULT 'active'
        )
        """
    )
    
    # Crear índices para mejorar performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_namespace ON chunks(namespace)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_namespace ON documents(namespace)")
    
    conn.commit()
    return conn

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 50) -> List[str]:
    """
    Divide el texto en chunks con overlap para mantener contexto
    
    Args:
        text: Texto a dividir
        chunk_size: Tamaño del chunk en tokens (palabras)
        overlap: Número de tokens de overlap entre chunks
    
    Returns:
        Lista de chunks de texto
    """
    if not text.strip():
        return []
    
    tokens = text.split()
    chunks = []
    i = 0
    
    while i < len(tokens):
        # Tomar chunk_size tokens desde la posición i
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = " ".join(chunk_tokens)
        
        # Solo agregar chunks no vacíos
        if chunk_text.strip():
            chunks.append(chunk_text)
        
        # Avanzar chunk_size - overlap tokens
        i += chunk_size - overlap
        
        # Evitar chunks muy pequeños al final
        if i >= len(tokens):
            break
    
    return chunks

def _extract_text_from_pdf_obj(pdf_obj) -> str:
    """
    Extrae texto de un objeto PDF (ruta o BytesIO)
    
    Args:
        pdf_obj: Ruta del archivo PDF o objeto BytesIO
    
    Returns:
        Texto extraído del PDF
    """
    try:
        reader = PdfReader(pdf_obj)
        full_text = ""
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n--- Página {page_num + 1} ---\n"
                    full_text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error extrayendo texto de página {page_num + 1}: {str(e)}")
                continue
        
        return full_text.strip()
    except Exception as e:
        logger.error(f"Error extrayendo texto del PDF: {str(e)}")
        raise ValueError(f"No se pudo extraer texto del PDF: {str(e)}")

def add_documents_from_pdf_path(pdf_path: str, namespace: str, description: Optional[str] = None):
    """
    Procesa un PDF desde ruta del archivo y lo agrega a la base vectorial
    
    Args:
        pdf_path: Ruta al archivo PDF
        namespace: Namespace para organizar el documento
        description: Descripción opcional del documento
    
    Returns:
        Dict con información del procesamiento
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Archivo no encontrado: {pdf_path}")
    
    # Obtener información del archivo
    file_size = os.path.getsize(pdf_path)
    filename = os.path.basename(pdf_path)
    
    # Extraer texto
    full_text = _extract_text_from_pdf_obj(pdf_path)
    
    # Calcular hash del contenido
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    
    return _index_text_chunks(
        full_text, 
        namespace, 
        filename, 
        file_size=file_size,
        content_hash=content_hash,
        description=description
    )

def add_documents_from_pdf_bytes(
    pdf_bytes: Union[bytes, bytearray], 
    namespace: str, 
    source: str = "<in-memory>",
    description: Optional[str] = None
):
    """
    Procesa un PDF desde bytes y lo agrega a la base vectorial
    
    Args:
        pdf_bytes: Contenido del PDF en bytes
        namespace: Namespace para organizar el documento
        source: Nombre del archivo fuente
        description: Descripción opcional del documento
    
    Returns:
        Dict con información del procesamiento
    """
    pdf_obj = BytesIO(pdf_bytes)
    file_size = len(pdf_bytes)
    
    # Extraer texto
    full_text = _extract_text_from_pdf_obj(pdf_obj)
    
    # Calcular hash del contenido
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    
    return _index_text_chunks(
        full_text, 
        namespace, 
        source, 
        file_size=file_size,
        content_hash=content_hash,
        description=description
    )

def _index_text_chunks(
    full_text: str, 
    namespace: str, 
    source: str,
    file_size: Optional[int] = None,
    content_hash: Optional[str] = None,
    description: Optional[str] = None
):
    """
    Procesa texto completo en chunks y los almacena en la base de datos
    
    Args:
        full_text: Texto completo del documento
        namespace: Namespace para organizar
        source: Nombre del archivo fuente
        file_size: Tamaño del archivo en bytes
        content_hash: Hash del contenido para deduplicación
        description: Descripción opcional
    
    Returns:
        Dict con estadísticas del procesamiento
    """
    if not full_text.strip():
        raise ValueError("El documento no contiene texto extraíble")
    
    # Obtener configuración de chunking desde variables de entorno
    from app.core.config import settings
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap
    
    # Dividir en chunks
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=chunk_overlap)
    
    if not chunks:
        raise ValueError("No se pudieron crear chunks del documento")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Verificar si el documento ya existe (por hash de contenido)
        if content_hash:
            cursor.execute(
                "SELECT COUNT(*) FROM documents WHERE content_hash=? AND namespace=? AND status='active'",
                (content_hash, namespace)
            )
            if cursor.fetchone()[0] > 0:
                logger.info(f"Documento duplicado detectado: {source} en namespace {namespace}")
                return {
                    "status": "duplicate",
                    "indexed_chunks": 0,
                    "namespace": namespace,
                    "total_chunks": len(chunks),
                    "message": "Documento ya existe en la base de datos"
                }
        
        # Registrar documento
        doc_id = hashlib.sha256(f"{namespace}:{source}:{content_hash}".encode("utf-8")).hexdigest()
        cursor.execute(
            """INSERT OR REPLACE INTO documents 
               (id, filename, namespace, file_size, content_hash, description, status) 
               VALUES (?, ?, ?, ?, ?, ?, 'active')""",
            (doc_id, source, namespace, file_size, content_hash, description)
        )
        
        # Procesar chunks
        added = 0
        skipped = 0
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            # Crear ID único para el chunk
            chunk_id = hashlib.sha256(
                f"{namespace}:{source}:{i}:{chunk[:100]}".encode("utf-8")
            ).hexdigest()
            
            # Verificar si el chunk ya existe
            cursor.execute("SELECT 1 FROM chunks WHERE id=? AND active=1", (chunk_id,))
            if cursor.fetchone():
                skipped += 1
                continue
            
            # Insertar chunk
            cursor.execute(
                """INSERT OR REPLACE INTO chunks 
                   (id, content, namespace, source, active) 
                   VALUES (?, ?, ?, ?, 1)""",
                (chunk_id, chunk, namespace, source)
            )
            added += 1
        
        conn.commit()
        
        logger.info(f"Documento procesado: {source} -> {added} chunks nuevos, {skipped} existentes")
        
        return {
            "status": "success",
            "indexed_chunks": added,
            "skipped_chunks": skipped,
            "total_chunks": len(chunks),
            "namespace": namespace,
            "document_id": doc_id,
            "content_hash": content_hash
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error indexando chunks para {source}: {str(e)}")
        raise
    finally:
        conn.close()

def list_namespaces() -> List[str]:
    """
    Lista todos los namespaces activos en la base de datos
    
    Returns:
        Lista de nombres de namespaces
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT DISTINCT namespace FROM chunks WHERE active=1 AND namespace IS NOT NULL ORDER BY namespace"
        )
        rows = cursor.fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()

def get_chunk_count(namespace: str) -> int:
    """
    Obtiene el número de chunks activos en un namespace
    
    Args:
        namespace: Nombre del namespace
    
    Returns:
        Número de chunks activos
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE namespace=? AND active=1", 
            (namespace,)
        )
        return cursor.fetchone()[0]
    finally:
        conn.close()

def get_document_info(namespace: str) -> List[Dict]:
    """
    Obtiene información de todos los documentos en un namespace
    
    Args:
        namespace: Nombre del namespace
    
    Returns:
        Lista de diccionarios con información de documentos
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """SELECT d.filename, d.file_size, d.upload_date, d.description,
                      COUNT(c.id) as chunk_count
               FROM documents d
               LEFT JOIN chunks c ON d.filename = c.source AND d.namespace = c.namespace
               WHERE d.namespace=? AND d.status='active' AND c.active=1
               GROUP BY d.id
               ORDER BY d.upload_date DESC""",
            (namespace,)
        )
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()

def delete_document(namespace: str, filename: str) -> Dict:
    """
    Marca un documento y sus chunks como inactivos (soft delete)
    
    Args:
        namespace: Namespace del documento
        filename: Nombre del archivo a eliminar
    
    Returns:
        Dict con información de la eliminación
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Contar chunks que se van a desactivar
        cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE namespace=? AND source=? AND active=1",
            (namespace, filename)
        )
        chunk_count = cursor.fetchone()[0]
        
        if chunk_count == 0:
            return {
                "status": "not_found",
                "message": f"No se encontró el documento {filename} en namespace {namespace}"
            }
        
        # Desactivar chunks
        cursor.execute(
            "UPDATE chunks SET active=0, updated_at=CURRENT_TIMESTAMP WHERE namespace=? AND source=? AND active=1",
            (namespace, filename)
        )
        
        # Desactivar documento
        cursor.execute(
            "UPDATE documents SET status='deleted' WHERE namespace=? AND filename=?",
            (namespace, filename)
        )
        
        conn.commit()
        
        return {
            "status": "deleted",
            "chunks_deactivated": chunk_count,
            "namespace": namespace,
            "filename": filename
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error eliminando documento {filename}: {str(e)}")
        raise
    finally:
        conn.close()

def cleanup_inactive_chunks() -> Dict:
    """
    Limpia chunks inactivos de la base de datos (hard delete)
    
    Returns:
        Dict con estadísticas de limpieza
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Contar chunks inactivos
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE active=0")
        inactive_count = cursor.fetchone()[0]
        
        # Eliminar chunks inactivos
        cursor.execute("DELETE FROM chunks WHERE active=0")
        
        # Eliminar documentos marcados como eliminados
        cursor.execute("DELETE FROM documents WHERE status='deleted'")
        
        conn.commit()
        
        return {
            "status": "cleaned",
            "chunks_deleted": inactive_count,
            "message": f"Se eliminaron {inactive_count} chunks inactivos"
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error limpiando chunks inactivos: {str(e)}")
        raise
    finally:
        conn.close()

# Compatibilidad con versión anterior
def add_documents_from_pdf(pdf_input, namespace: str):
    """Función de compatibilidad para llamadas genéricas"""
    if isinstance(pdf_input, (bytes, bytearray)):
        return add_documents_from_pdf_bytes(pdf_input, namespace, source="<bytes>")
    elif isinstance(pdf_input, str):
        return add_documents_from_pdf_path(pdf_input, namespace)
    else:
        raise ValueError("Tipo no soportado para add_documents_from_pdf")