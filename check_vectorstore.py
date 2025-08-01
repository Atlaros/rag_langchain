#!/usr/bin/env python3
import os
import sys
import sqlite3
from dotenv import load_dotenv
from pathlib import Path

# cargar .env si existe
load_dotenv()

from app.vector_store.faiss_store import list_namespaces, DB_PATH
from app.langchain_wrapper import _get_or_build_vectorstore
from app.core.config import settings

def main():
    embedding_model = os.getenv("EMBEDDING_MODEL", settings.embedding_model)
    k = int(os.getenv("RETRIEVAL_K", settings.retrieval_k))
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "estrategias de mercado"

    print(f"Embedding model: {embedding_model}")
    namespaces = list_namespaces()
    print("Namespaces detected:", namespaces)

    # contar chunks por namespace
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = conn.cursor()
    for ns in namespaces:
        cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE namespace=? AND active=1", (ns,)
        )
        count = cursor.fetchone()[0]
        print(f"  - Namespace '{ns}': {count} active chunks")

    if not namespaces:
        print("No namespaces found, nada que probar.")
        return

    test_ns = namespaces[0]
    print(f"\nProbando recuperación en namespace '{test_ns}' para query: '{query}'")
    vectorstore = _get_or_build_vectorstore(test_ns, embedding_model, k)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    for i, d in enumerate(docs, 1):
        excerpt = getattr(d, "page_content", "")[:500].replace("\n", " ")
        print(f"\n--- Documento {i} ---")
        print("Metadata:", d.metadata)
        print("Extracto:", excerpt[:300] + ("..." if len(excerpt) > 300 else ""))

if __name__ == "__main__":
    main()
