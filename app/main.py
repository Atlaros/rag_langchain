import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time

from app.api.endpoints import router
from app.auto_ingest import scan_and_ingest
from app.core.config import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    logger.info("🚀 Iniciando RAG Microservice")
    
    # Tareas de inicialización
    try:
        # Ejecutar auto-ingesta en background
        asyncio.create_task(background_startup_tasks())
        logger.info("✅ Tareas de inicialización programadas")
    except Exception as e:
        logger.error(f"❌ Error en inicialización: {str(e)}")
    
    yield
    
    # Tareas de limpieza al cerrar
    logger.info("🛑 Cerrando RAG Microservice")

# Crear aplicación FastAPI
app = FastAPI(
    title="RAG Microservice",
    description="Microservicio para Retrieval-Augmented Generation con PDFs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# === MIDDLEWARE ===

# CORS para permitir solicitudes desde diferentes dominios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Compresión GZIP para respuestas
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} - Client: {request.client.host}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"📤 {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Agregar header de tiempo de procesamiento
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"❌ {request.method} {request.url.path} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.3f}s"
        )
        raise

# === MANEJADORES DE ERRORES ===

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Manejador para errores HTTP"""
    logger.warning(f"HTTP Error {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Manejador para errores de validación de datos"""
    logger.warning(f"Validation Error: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "Error de validación en los datos enviados",
                "type": "validation_error",
                "details": exc.errors()
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Manejador general para errores no controlados"""
    logger.error(f"Unhandled Error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Error interno del servidor",
                "type": "internal_error"
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# === ENDPOINTS RAÍZ ===

@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "RAG Microservice",
        "version": "1.0.0",
        "description": "Microservicio para Retrieval-Augmented Generation con PDFs",
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
        "endpoints": {
            "upload_document": "POST /documents/upload",
            "query": "POST /query",
            "batch_query": "POST /query/batch", 
            "list_namespaces": "GET /namespaces",
            "rebuild_vectorstore": "POST /vectorstore/rebuild",
            "sync_documents": "POST /documents/sync"
        },
        "models": {
            "embedding": settings.embedding_model,
            "generation": settings.generation_model
        }
    }

# === INCLUIR RUTAS ===

# Incluir todas las rutas de la API
app.include_router(router, prefix="/api/v1", tags=["RAG API"])

# También incluir rutas principales sin prefijo para compatibilidad
app.include_router(router)

# === FUNCIONES DE INICIALIZACIÓN ===

async def background_startup_tasks():
    """Tareas que se ejecutan en background al iniciar"""
    try:
        logger.info("🔄 Iniciando tareas de background...")
        
        # Esperar un poco para que la app esté completamente lista
        await asyncio.sleep(2)
        
        # Ejecutar auto-ingesta de PDFs
        logger.info("📚 Ejecutando auto-ingesta de documentos...")
        result = scan_and_ingest()
        logger.info(f"✅ Auto-ingesta completada: {result}")
        
        # Verificar configuración
        await verify_configuration()
        
        logger.info("🎉 Todas las tareas de inicialización completadas")
        
    except Exception as e:
        logger.error(f"❌ Error en tareas de background: {str(e)}")

async def verify_configuration():
    """Verifica que la configuración esté correcta"""
    try:
        # Verificar token de Hugging Face
        if not settings.hf_token:
            logger.warning("⚠️  Token de Hugging Face no configurado")
        
        # Verificar modelos
        logger.info(f"🤖 Modelo de embeddings: {settings.embedding_model}")
        logger.info(f"🧠 Modelo de generación: {settings.generation_model}")
        
        # Verificar base de datos
        from app.vector_store.faiss_store import get_connection, list_namespaces
        conn = get_connection()
        conn.close()
        
        namespaces = list_namespaces()
        logger.info(f"📂 Namespaces disponibles: {namespaces}")
        
        logger.info("✅ Configuración verificada correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error verificando configuración: {str(e)}")

# === INFORMACIÓN ADICIONAL ===

if __name__ == "__main__":
    import uvicorn
    
    # Configuración para desarrollo
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

