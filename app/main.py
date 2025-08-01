from fastapi import FastAPI
from app.api.endpoints import router
from app.auto_ingest import scan_and_ingest
import asyncio

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup_tasks():
    asyncio.create_task(_background_ingest())

async def _background_ingest():
    try:
        scan_and_ingest()
    except Exception as e:
        print("[auto_ingest error]", e)


