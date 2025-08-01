import os
from pathlib import Path

from app.core.config import settings
from app.vector_store.faiss_store import get_connection, list_namespaces

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# raíces
PROJECT_ROOT = Path(__file__).resolve().parent
VECTORSTORE_ROOT = PROJECT_ROOT.parent / "vectorstores"

def _load_chunks(namespace: str | None = None):
    conn = get_connection()
    cursor = conn.cursor()
    if namespace:
        cursor.execute(
            "SELECT content, namespace, source FROM chunks WHERE namespace=? AND active=1",
            (namespace,),
        )
    else:
        cursor.execute("SELECT content, namespace, source FROM chunks WHERE active=1")
    rows = cursor.fetchall()
    texts = []
    metadatas = []
    for content, ns, source in rows:
        texts.append(content)
        meta = {"namespace": ns, "source": source, "doc_id": source}
        metadatas.append(meta)
    return texts, metadatas

def _get_or_build_vectorstore(namespace: str, embedding_model_name: str, k: int):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vs_dir = VECTORSTORE_ROOT / namespace

    if vs_dir.exists():
        try:
            vectorstore = FAISS.load_local(str(vs_dir), embeddings)
            return vectorstore
        except Exception:
            pass  # reconstruir si falla

    texts, metadatas = _load_chunks(namespace)
    if not texts:
        vectorstore = FAISS.from_texts([], embeddings, metadatas=[])
    else:
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    vs_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vs_dir))
    return vectorstore

def build_and_persist_vectorstores_for_all_namespaces():
    embedding_model = os.getenv("EMBEDDING_MODEL", settings.embedding_model)
    namespaces = list_namespaces()
    for namespace in namespaces:
        _get_or_build_vectorstore(namespace, embedding_model, settings.retrieval_k)

def build_retrieval_chain(
    namespace: str | None,
    system_prompt: str,
    question: str,
    k: int,
    temperature: float,
    max_tokens: int,
):
    embedding_model = os.getenv("EMBEDDING_MODEL", settings.embedding_model)
    generation_model = os.getenv("GENERATION_MODEL", settings.generation_model)

    if not namespace:
        namespace = ""  # o usar "compromidos" por defecto si corresponde

    vectorstore = _get_or_build_vectorstore(namespace, embedding_model, k)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = HuggingFaceHub(
        repo_id=generation_model,
        task="text2text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
    )

    template = """
Sos un asistente experto. Usar el siguiente contexto para responder.

Contexto:
{context}

PREGUNTA:
{question}

Responde con diagnóstico breve, recomendaciones y ejemplos. Si falta información, indícalo claramente.
""".strip()

    prompt = PromptTemplate(
        input_variables=["context", "question"], template=template
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_prompt=prompt,
    )
    return chain


