import json
from pathlib import Path
from app.vector_store.faiss_store import add_documents_from_pdf_path, list_namespaces
from app.langchain_wrapper import build_and_persist_vectorstores_for_all_namespaces

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT.parent / "pdfs"
STATE_PATH = PDF_DIR / ".ingest_state.json"

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except:
            return {}
    return {}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))

def scan_and_ingest():
    PDF_DIR.mkdir(exist_ok=True)
    state = load_state()
    changed = False

    for pdf in PDF_DIR.glob("*.pdf"):
        mtime = pdf.stat().st_mtime
        key = str(pdf.name)
        if key not in state or state[key] != mtime:
            namespace = "compromidos"  # conservás el namespace existente
            print(f"[auto_ingest] Ingestando {pdf.name} en namespace '{namespace}'")
            add_documents_from_pdf_path(str(pdf), namespace=namespace)
            state[key] = mtime
            changed = True

    if changed:
        save_state(state)

    build_and_persist_vectorstores_for_all_namespaces()
    return {"scanned": len(list(PDF_DIR.glob("*.pdf"))), "updated": changed}
