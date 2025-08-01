# 🚀 RAG Microservicio

Microservicio completo para **Retrieval-Augmented Generation (RAG)** que permite subir documentos PDF, crear bases vectoriales automáticamente y realizar consultas inteligentes usando modelos de Hugging Face.

## 🎯 Características Principales

- **📄 Procesamiento de PDFs**: Carga y procesa documentos PDF automáticamente
- **🔍 Base Vectorial**: Crea y mantiene índices vectoriales usando FAISS
- **🤖 IA Multilingüe**: Soporte para consultas en múltiples idiomas
- **📊 API REST Completa**: Endpoints JSON para todas las operaciones
- **🏷️ Namespaces**: Organización de documentos por categorías
- **⚡ Consultas en Lote**: Procesamiento de múltiples consultas simultáneas
- **🔄 Auto-sincronización**: Detección automática de nuevos PDFs
- **📈 Monitoreo**: Health checks y métricas de estado
- **🐳 Docker Ready**: Containerizado y listo para producción

## 🛠️ Tecnologías

- **FastAPI**: Framework web moderno y rápido
- **LangChain**: Orquestación de modelos de IA
- **Hugging Face**: Modelos de embeddings y generación
- **FAISS**: Búsqueda vectorial eficiente
- **SQLite**: Base de datos para metadatos
- **Docker**: Containerización para despliegue

## 🚀 Inicio Rápido

### 1. Configuración del Entorno

```bash
# Clonar el repositorio
git clone <repository-url>
cd rag-microservice

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o en Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración

```bash
# Copiar template de configuración
cp .env.template .env

# Editar .env y agregar tu token de Hugging Face
HUGGINGFACEHUB_API_TOKEN=tu_token_aqui
```

### 3. Ejecutar el Microservicio

```bash
# Modo desarrollo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# O usar el script de inicio
./scripts/start.sh  # Linux/Mac
# o en Windows: .\scripts\start.ps1
```

### 4. Con Docker

```bash
# Construir y ejecutar
docker-compose up --build

# Solo ejecutar (si ya está construido)
docker-compose up -d
```

## 📋 API Endpoints

### 🔍 Estado y Salud

```http
GET /health
GET /status
GET /namespaces
```

### 📄 Gestión de Documentos

```http
# Subir documento PDF
POST /documents/upload
Content-Type: multipart/form-data

{
  "file": <archivo_pdf>,
  "namespace": "mi_namespace",
  "description": "Descripción opcional"
}
```

```http
# Sincronizar PDFs desde directorio
POST /documents/sync
```

### 🤖 Consultas RAG

```http
# Consulta simple
POST /query
Content-Type: application/json

{
  "query": "¿Cuáles son los puntos principales?",
  "namespace": "compromidos",
  "max_results": 5,
  "temperature": 0.7,
  "max_tokens": 600
}
```

```http
# Consultas en lote
POST /query/batch
Content-Type: application/json

{
  "queries": [
    "Primera pregunta",
    "Segunda pregunta"
  ],
  "namespace": "compromidos",
  "max_results": 3
}
```

### ⚙️ Administración

```http
# Reconstruir índice vectorial
POST /vectorstore/rebuild?namespace=mi_namespace
```

## 📊 Ejemplos de Uso

### Subir un Documento

```python
import requests

# Subir PDF
with open('documento.pdf', 'rb') as f:
    files = {'file': ('documento.pdf', f, 'application/pdf')}
    data = {'namespace': 'documentos', 'description': 'Mi documento'}
    
    response = requests.post(
        'http://localhost:8000/documents/upload',
        files=files,
        data=data
    )
    
print(response.json())
```

### Realizar Consulta

```python
import requests

# Consulta RAG
query_data = {
    "query": "Resume los puntos más importantes",
    "namespace": "documentos",
    "max_results": 5,
    "temperature": 0.7
}

response = requests.post(
    'http://localhost:8000/query',
    json=query_data
)

result = response.json()
print(f"Respuesta: {result['answer']}")
print(f"Fuentes utilizadas: {len(result['sources_used'])}")
```

### Consultas en Lote

```python
import requests

# Múltiples consultas
batch_data = {
    "queries": [
        "¿Cuál es el tema principal?",
        "¿Qué recomendaciones se mencionan?",
        "Resume en 3 puntos clave"
    ],
    "namespace": "documentos",
    "max_results": 3
}

response = requests.post(
    'http://localhost:8000/query/batch',
    json=batch_data
)

results = response.json()
for i, result in enumerate(results['results']):
    print(f"Consulta {i+1}: {result['answer'][:100]}...")
```

## ⚙️ Configuración Avanzada

### Variables de Entorno Principales

```bash
# Modelos de IA
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
GENERATION_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Configuración de chunks
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Parámetros de generación
TEMPERATURE=0.7
MAX_TOKENS=600
RETRIEVAL_K=5
```

### Modelos Recomendados

#### Para Desarrollo (Más Rápido)
```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GENERATION_MODEL=google/flan-t5-small
```

#### Para Producción (Más Preciso)
```bash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
GENERATION_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

## 🧪 Pruebas

### Ejecutar Suite de Pruebas

```bash
# Pruebas completas
python test_microservice.py

# Solo pruebas de API JSON
python test_microservice.py --json-only

# Pruebas con URL personalizada
python test_microservice.py --url http://mi-servidor:8000
```

### Verificar Funcionamiento

```bash
# Health check
curl http://localhost:8000/health

# Estado del servicio
curl http://localhost:8000/status

# Listar namespaces
curl http://localhost:8000/namespaces
```

## 🐳 Despliegue en Producción

### Docker Compose

```bash
# Configurar variables de producción
cp .env.template .env
# Editar .env con configuración de producción

# Desplegar
docker-compose -f docker-compose.production.yml up -d
```

### Con Reverse Proxy (Nginx)

```bash
# Activar perfil de producción con proxy
docker-compose --profile production up -d
```

### Recursos Recomendados

- **CPU**: Mínimo 2 cores, recomendado 4+ cores
- **RAM**: Mínimo 4GB, recomendado 8GB+ para modelos grandes
- **Almacenamiento**: 10GB+ para modelos y datos

## 📁 Estructura del Proyecto

```
rag-microservice/
├── app/
│   ├── api/
│   │   └── endpoints.py      # Endpoints de la API
│   ├── core/
│   │   ├── config.py         # Configuración
│   │   └── prompt_manager.py # Gestión de prompts
│   ├── vector_store/
│   │   └── faiss_store.py    # Almacén vectorial
│   ├── langchain_wrapper.py  # Integración LangChain
│   ├── auto_ingest.py        # Auto-ingesta de PDFs
│   └── main.py               # Aplicación principal
├── pdfs/                     # Directorio de PDFs
├── vectorstores/             # Índices vectoriales FAISS
├── scripts/                  # Scripts de utilidad
├── docker-compose.yml        # Docker para desarrollo
├── docker-compose.production.yml # Docker para producción
├── requirements.txt          # Dependencias Python
├── test_microservice.py      # Suite de pruebas
└── README.md                 # Esta documentación
```

## 🔧 Troubleshooting

### Problemas Comunes

#### Error de Token de Hugging Face
```bash
# Verificar que el token esté configurado
echo $HUGGINGFACEHUB_API_TOKEN

# O verificar en el archivo .env
grep HUGGINGFACEHUB_API_TOKEN .env
```

#### Memoria Insuficiente
```bash
# Usar modelos más ligeros
GENERATION_MODEL=google/flan-t5-small
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### PDFs No Se Procesan
```bash
# Verificar directorio de PDFs
ls -la pdfs/

# Forzar sincronización
curl -X POST http://localhost:8000/documents/sync
```

#### Base Vectorial Corrupta
```bash
# Reconstruir índices
curl -X POST http://localhost:8000/vectorstore/rebuild

# O eliminar y reconstruir
rm -rf vectorstores/
curl -X POST http://localhost:8000/vectorstore/rebuild
```

### Logs y Depuración

```bash
# Ver logs en tiempo real
docker-compose logs -f rag-microservice

# Ver logs específicos
docker logs rag-microservice

# Verificar base de datos
python inspect_chunks.py
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## 🙋‍♂️ Soporte

- **Documentación**: `/docs` (FastAPI auto-docs)
- **Issues**: GitHub Issues
- **API Reference**: `/redoc`

---

**¿Necesitas ayuda?** Abre un issue o consulta la documentación automática en `/docs` una vez que el servicio esté ejecutándose.