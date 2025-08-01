import os
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.vector_store.faiss_store import add_documents_from_pdf_bytes, list_namespaces, get_chunk_count
from app.langchain_wrapper import build_retrieval_chain, build_and_persist_vectorstores_for_all_namespaces
from app.auto_ingest import scan_and_ingest

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# === MODELOS DE REQUEST/RESPONSE ===

class DocumentUploadRequest(BaseModel):
    namespace: str = Field(..., description="Namespace para organizar documentos")
    description: Optional[str] = Field(None, description="Descripción opcional del documento")

class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    namespace: str
    chunks_added: int
    total_chunks_in_namespace: int
    doc_id: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Pregunta o consulta a realizar")
    namespace: str = Field(default="compromidos", description="Namespace donde buscar")
    max_results: Optional[int] = Field(default=5, ge=1, le=20, description="Máximo número de documentos a recuperar")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperatura para generación")
    max_tokens: Optional[int] = Field(default=600, ge=50, le=2000, description="Máximo tokens en respuesta")

class DocumentReference(BaseModel):
    doc_id: str
    source: str
    relevance_score: float
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    query: str
    namespace: str
    confidence: str
    sources_used: List[DocumentReference]
    metadata: Dict[str, Any]

class NamespaceInfo(BaseModel):
    namespace: str
    document_count: int
    chunk_count: int

class StatusResponse(BaseModel):
    status: str
    service: str
    version: str
    namespaces: List[NamespaceInfo]
    models: Dict[str, str]

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=10)
    namespace: str = Field(default="compromidos")
    max_results: Optional[int] = Field(default=3, ge=1, le=10)

class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]
    total_queries: int
    namespace: str

# === ENDPOINTS PRINCIPALES ===

@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint para verificar que el servicio está funcionando"""
    return {
        "status": "healthy", 
        "service": "RAG Microservice",
        "timestamp": str(pd.Timestamp.now())
    }

@router.get("/status", response_model=StatusResponse)
async def get_service_status():
    """Obtiene el estado completo del servicio incluyendo namespaces y modelos"""
    try:
        namespaces = list_namespaces()
        namespace_info = []
        
        for ns in namespaces:
            chunk_count = get_chunk_count(ns)
            # Contar documentos únicos por fuente
            from app.vector_store.faiss_store import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(DISTINCT source) FROM chunks WHERE namespace=? AND active=1", 
                (ns,)
            )
            doc_count = cursor.fetchone()[0]
            
            namespace_info.append(NamespaceInfo(
                namespace=ns,
                document_count=doc_count,
                chunk_count=chunk_count
            ))
        
        return StatusResponse(
            status="operational",
            service="RAG Microservice",
            version="1.0.0",
            namespaces=namespace_info,
            models={
                "embedding": settings.embedding_model,
                "generation": settings.generation_model
            }
        )
    except Exception as e:
        logger.error(f"Error getting service status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting service status: {str(e)}")

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Sube un documento PDF y lo procesa para crear/actualizar la base vectorial
    """
    # Validaciones
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB límite
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máximo 50MB)")
    
    try:
        # Leer contenido del archivo
        pdf_bytes = await file.read()
        logger.info(f"Procesando documento: {file.filename} ({len(pdf_bytes)} bytes)")
        
        # Procesar PDF y agregar a base vectorial
        result = add_documents_from_pdf_bytes(
            pdf_bytes, 
            namespace, 
            source=file.filename
        )
        
        # Obtener conteo total de chunks en el namespace
        total_chunks = get_chunk_count(namespace)
        
        # Reconstruir índice vectorial en background
        background_tasks.add_task(rebuild_vectorstore_for_namespace, namespace)
        
        return DocumentUploadResponse(
            status="success",
            message=f"Documento procesado exitosamente",
            namespace=namespace,
            chunks_added=result["indexed_chunks"],
            total_chunks_in_namespace=total_chunks,
            doc_id=file.filename
        )
        
    except Exception as e:
        logger.error(f"Error procesando documento {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Realiza una consulta RAG sobre los documentos en el namespace especificado
    """
    try:
        logger.info(f"Consulta RAG: '{request.query}' en namespace '{request.namespace}'")
        
        # Verificar que el namespace existe y tiene documentos
        chunk_count = get_chunk_count(request.namespace)
        if chunk_count == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontraron documentos en el namespace '{request.namespace}'"
            )
        
        # Construir cadena de recuperación
        chain = build_retrieval_chain(
            namespace=request.namespace,
            system_prompt=settings.system_prompt,
            question=request.query,
            k=request.max_results,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Ejecutar consulta
        result = chain({"query": request.query})
        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])
        
        # Procesar documentos fuente
        sources = []
        for i, doc in enumerate(source_docs):
            metadata = getattr(doc, "metadata", {})
            content = getattr(doc, "page_content", "")
            
            sources.append(DocumentReference(
                doc_id=metadata.get("doc_id", f"doc_{i}"),
                source=metadata.get("source", "unknown"),
                relevance_score=0.9 - (i * 0.1),  # Score simulado basado en orden
                excerpt=content[:300].replace("\n", " ") + ("..." if len(content) > 300 else "")
            ))
        
        # Determinar confianza basada en número de fuentes
        confidence = "high" if len(sources) >= 3 else "medium" if len(sources) >= 1 else "low"
        
        return QueryResponse(
            answer=answer,
            query=request.query,
            namespace=request.namespace,
            confidence=confidence,
            sources_used=sources,
            metadata={
                "model_used": settings.generation_model,
                "chunks_retrieved": len(sources),
                "total_chunks_available": chunk_count,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en consulta RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando consulta: {str(e)}")

@router.post("/query/batch", response_model=BatchQueryResponse)
async def batch_query_documents(request: BatchQueryRequest):
    """
    Realiza múltiples consultas RAG en lote
    """
    try:
        results = []
        
        for query in request.queries:
            query_request = QueryRequest(
                query=query,
                namespace=request.namespace,
                max_results=request.max_results
            )
            
            try:
                result = await query_documents(query_request)
                results.append(result)
            except Exception as e:
                # En caso de error en una consulta, agregar resultado de error
                results.append(QueryResponse(
                    answer=f"Error procesando consulta: {str(e)}",
                    query=query,
                    namespace=request.namespace,
                    confidence="error",
                    sources_used=[],
                    metadata={"error": str(e)}
                ))
        
        return BatchQueryResponse(
            results=results,
            total_queries=len(request.queries),
            namespace=request.namespace
        )
        
    except Exception as e:
        logger.error(f"Error en consulta batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando consultas en lote: {str(e)}")

@router.get("/namespaces", response_model=List[NamespaceInfo])
async def list_document_namespaces():
    """
    Lista todos los namespaces disponibles con información de documentos
    """
    try:
        namespaces = list_namespaces()
        namespace_info = []
        
        for ns in namespaces:
            chunk_count = get_chunk_count(ns)
            
            # Contar documentos únicos
            from app.vector_store.faiss_store import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(DISTINCT source) FROM chunks WHERE namespace=? AND active=1", 
                (ns,)
            )
            doc_count = cursor.fetchone()[0]
            
            namespace_info.append(NamespaceInfo(
                namespace=ns,
                document_count=doc_count,
                chunk_count=chunk_count
            ))
        
        return namespace_info
        
    except Exception as e:
        logger.error(f"Error listando namespaces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo namespaces: {str(e)}")

@router.post("/vectorstore/rebuild")
async def rebuild_vectorstore(
    background_tasks: BackgroundTasks,
    namespace: Optional[str] = None
):
    """
    Reconstruye el índice vectorial para un namespace específico o todos
    """
    try:
        if namespace:
            background_tasks.add_task(rebuild_vectorstore_for_namespace, namespace)
            message = f"Reconstrucción del índice vectorial iniciada para namespace '{namespace}'"
        else:
            background_tasks.add_task(build_and_persist_vectorstores_for_all_namespaces)
            message = "Reconstrucción de todos los índices vectoriales iniciada"
        
        return JSONResponse(content={
            "status": "accepted",
            "message": message,
            "note": "El proceso se ejecuta en segundo plano"
        })
        
    except Exception as e:
        logger.error(f"Error iniciando reconstrucción de vectorstore: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error iniciando reconstrucción: {str(e)}")

@router.post("/documents/sync")
async def sync_pdf_documents(background_tasks: BackgroundTasks):
    """
    Sincroniza documentos PDF desde el directorio /pdfs
    """
    try:
        background_tasks.add_task(scan_and_ingest)
        
        return JSONResponse(content={
            "status": "accepted",
            "message": "Sincronización de documentos PDF iniciada",
            "note": "El proceso se ejecuta en segundo plano"
        })
        
    except Exception as e:
        logger.error(f"Error iniciando sincronización: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error iniciando sincronización: {str(e)}")

# === FUNCIONES AUXILIARES ===

async def rebuild_vectorstore_for_namespace(namespace: str):
    """Función auxiliar para reconstruir vectorstore en background"""
    try:
        from app.langchain_wrapper import _get_or_build_vectorstore
        embedding_model = os.getenv("EMBEDDING_MODEL", settings.embedding_model)
        _get_or_build_vectorstore(namespace, embedding_model, settings.retrieval_k)
        logger.info(f"Vectorstore reconstruido para namespace: {namespace}")
    except Exception as e:
        logger.error(f"Error reconstruyendo vectorstore para {namespace}: {str(e)}")

# === IMPORTS ADICIONALES ===
import pandas as pd
