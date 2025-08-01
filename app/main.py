from fastapi import FastAPI
from app.api.endpoints import router
from app.auto_ingest import scan_and_ingest

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup_sync_pdfs():
    scan_and_ingest()
