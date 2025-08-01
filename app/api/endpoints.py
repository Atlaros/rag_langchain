import os
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from app.core.config import settings
from app.core.prompt_manager import PromptManager
from app.vector_store.faiss_store import add_documents_from_pdf
from app.langchain_wrapper import build_retrieval_chain
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.vector_store.faiss_store import add_documents_from_pdf_bytes


router = APIRouter()
prompt_mgr = PromptManager()

class RagRequest(BaseModel):
    type: str
    user_id: Optional[str] = None
    role: Optional[str] = None
    email_text: Optional[str] = None
    brief: Optional[str] = None

    def get_query_text(self):
        return self.email_text if self.type == 'query' else self.brief or ''

class DocumentMeta(BaseModel):
    doc_id: str
    score: float
    excerpt: str

class RagResponse(BaseModel):
    answer: str
    type: str
    used_documents: List[DocumentMeta]
    meta: dict

@router.get('/healthz')
async def healthz():
    return {'status':'ok', 'timestamp': __import__('datetime').datetime.utcnow().isoformat()}

@router.post("/ingest_pdf")
async def ingest_pdf(namespace: str = Form(...), file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    try:
        pdf_bytes = await file.read()
        result = add_documents_from_pdf_bytes(pdf_bytes, namespace, source=file.filename)
        return {"status": "success", **result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Aquí puedes loguear e para debug si quieres
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")

@router.post('/rag', response_model=RagResponse)
async def rag_endpoint(request: RagRequest):
    if request.type not in ('query', 'campaign'):
        raise HTTPException(status_code=400, detail='Invalid type')
    namespace = request.user_id or 'default'
    k = settings.retrieval_k
    system = settings.system_prompt
    chain = build_retrieval_chain(namespace, system_prompt=system, k=k)
    question = request.get_query_text()
    result = chain({"question": question})
    answer = result.get('result') or ''
    source_docs = result.get('source_documents', [])
    used_docs = []
    for doc in source_docs:
        meta = getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else {}
        excerpt = getattr(doc, 'page_content', '')[:500].replace('\n',' ') if hasattr(doc, 'page_content') else ''
        used_docs.append(DocumentMeta(
            doc_id=meta.get('doc_id','') or meta.get('source',''),
            score=0.0,
            excerpt=excerpt or (doc.content[:500] if hasattr(doc,'content') else '')
        ))
    return RagResponse(
        answer=answer.strip(),
        type=request.type,
        used_documents=used_docs,
        meta={
            'model': os.getenv('GENERATION_MODEL', 'unknown'),
            'tokens_used': None
        }
    )

app = FastAPI()
app.include_router(router)
