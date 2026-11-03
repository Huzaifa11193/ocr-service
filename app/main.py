from fastapi import FastAPI

from .routers import health, ocr

app = FastAPI(title="DocIntel OCR Service", version="0.2.0")
app.include_router(health.router)
app.include_router(ocr.router)
