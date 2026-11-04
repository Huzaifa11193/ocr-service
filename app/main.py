from fastapi import FastAPI

from .routers import health, ocr, transcribe

app = FastAPI(title="DocIntel OCR Service", version="0.3.0")
app.include_router(health.router)
app.include_router(ocr.router)
app.include_router(transcribe.router)
