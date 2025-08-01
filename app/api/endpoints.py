import os
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.prompt_manager import PromptManager
from app.vector_store.faiss_store import add_documents_from_pdf_path, add_documents_from_pdf_bytes
from app.langchain_wrapper import build_retrieval_chain
from app.auto_ingest import scan_and_ingest

router = APIRouter()
prompt_mgr = PromptManager()

class RagRequest(BaseModel):
    type: str
    user_id: Optional[str] = None
    role: Optional[str] = None
    email_text: Optional[str] = None
    brief: Optional[str] = None

class DocumentMeta(BaseModel):
    doc_id: str
    score: float
    excerpt: str

class RagResponse(BaseModel):
    answer: str
    type: str
    used_documents: List[DocumentMeta]
    meta: dict

@router.get("/healthz")
async def healthz():
    return {"status": "ok"}

@router.post("/ingest_pdf")
async def ingest_pdf(namespace: str = Form(...), file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    try:
        pdf_bytes = await file.read()
        res = add_documents_from_pdf_bytes(pdf_bytes, namespace, source=file.filename)
        # después de ingestar, reconstruir índice de ese namespace
        from app.langchain_wrapper import _get_or_build_vectorstore  # internal
        embedding_model = os.getenv("EMBEDDING_MODEL", settings.embedding_model)
        _get_or_build_vectorstore(namespace, embedding_model, settings.retrieval_k)
        return JSONResponse(content={"status": "ingested", **res})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag", response_model=RagResponse)
async def rag_endpoint(request: RagRequest):
    if request.type not in ("query", "campaign"):
        raise HTTPException(status_code=400, detail="Invalid type")
    namespace = request.user_id or "compromidos"
    question = request.email_text if request.type == "query" else request.brief or ""
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")

    # parámetros desde settings/env
    k = settings.retrieval_k
    temperature = settings.temperature
    max_tokens = settings.max_tokens

    chain = build_retrieval_chain(
        namespace=namespace,
        system_prompt="",  # si usás alguno más arriba podés pasarlo
        question=question,
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result = chain({"query": question})
    answer = result.get("result") or ""
    source_docs = result.get("source_documents", [])
    used_docs = []
    for doc in source_docs:
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        excerpt = getattr(doc, "page_content", "")[:500].replace("\n", " ") if hasattr(doc, "page_content") else ""
        used_docs.append(
            DocumentMeta(
                doc_id=meta.get("doc_id", "") or meta.get("source", ""),
                score=0.0,
                excerpt=excerpt,
            )
        )
    return RagResponse(
        answer=answer.strip(),
        type=request.type,
        used_documents=used_docs,
        meta={
            "model": os.getenv("GENERATION_MODEL", "unknown"),
            "tokens_used": None,
        },
    )

@router.post("/sync_pdfs")
async def sync_pdfs():
    result = scan_and_ingest()
    return JSONResponse(content={"status": "synced", **result})
