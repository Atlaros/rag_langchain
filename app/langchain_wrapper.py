import os
import sqlite3
from app.core.config import settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

DB_PATH = "vector_metadata.db"

def _load_chunks(namespace: str | None = None):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    if namespace:
        cursor.execute("SELECT content, id FROM chunks WHERE active=1 AND namespace=?", (namespace,))
    else:
        cursor.execute("SELECT content, id FROM chunks WHERE active=1")
    rows = cursor.fetchall()
    texts = [r[0] for r in rows]
    metadatas = [{"doc_id": r[1], "namespace": namespace or ""} for r in rows]
    return texts, metadatas

def build_retrieval_chain(namespace: str, system_prompt: str, k: int = 5):
    embedding_model = os.getenv("EMBEDDING_MODEL", settings.embedding_model)
    generation_model = os.getenv("GENERATION_MODEL", settings.generation_model)
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face API token not set in environment as HUGGINGFACEHUB_API_TOKEN or HUGGINGFACE_API_TOKEN")
    texts, metadatas = _load_chunks(namespace)
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    if not texts:
        vectorstore = FAISS.from_texts([], embeddings, metadatas=[])
    else:
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # LLM
    llm = HuggingFaceHub(repo_id=generation_model, model_kwargs={"temperature": float(os.getenv('TEMPERATURE', settings.temperature)), "max_new_tokens": int(os.getenv('MAX_TOKENS', settings.max_tokens))}, huggingfacehub_api_token=hf_token)
    template = """{system_prompt}

CONTEXTO RELEVANTE:
{context}

PREGUNTA:
{question}

Responde con diagnóstico breve, recomendaciones y ejemplos. Si falta información, indícalo claramente.""".strip()
    prompt = PromptTemplate(input_variables=["system_prompt", "context", "question"], template=template)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_prompt=prompt
    )
    return chain
