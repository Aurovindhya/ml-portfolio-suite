"""
gateway/main.py

FastAPI gateway — single entry point for all modules.
Mounts each module's router under /api/v1/{domain}/{module}.

Run:
    uvicorn gateway.main:app --reload
    Open http://localhost:8000/docs for Swagger UI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import API_PREFIX
from core.response import HealthStatus

# ── Module routers ────────────────────────────────────────────────────────────
from modules.regression.flight_fare.router   import router as flight_router
from modules.regression.house_prices.router  import router as house_router
from modules.regression.gold_prices.router   import router as gold_router
from modules.classification.heart_disease.router import router as heart_router
from modules.classification.spam_filter.router   import router as spam_router
from modules.classification.chest_disease.router import router as chest_router
from modules.generative.chatbot.router       import router as chatbot_router
from modules.generative.pdf_assistant.router import router as pdf_router
from modules.generative.medical_tool.router  import router as medical_router
from modules.computer_vision.hand_tracking.router import router as vision_router
from modules.agentic.market_insights.router  import router as market_router

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Portfolio Suite",
    description=(
        "A full-stack Machine Learning platform spanning classical ML, deep learning, "
        "Generative AI, computer vision, and agentic workflows — unified behind a single "
        "FastAPI orchestration layer."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(flight_router,  prefix=f"{API_PREFIX}/regression",      tags=["Regression"])
app.include_router(house_router,   prefix=f"{API_PREFIX}/regression",      tags=["Regression"])
app.include_router(gold_router,    prefix=f"{API_PREFIX}/regression",      tags=["Regression"])
app.include_router(heart_router,   prefix=f"{API_PREFIX}/classification",  tags=["Classification"])
app.include_router(spam_router,    prefix=f"{API_PREFIX}/classification",  tags=["Classification"])
app.include_router(chest_router,   prefix=f"{API_PREFIX}/classification",  tags=["Classification"])
app.include_router(chatbot_router, prefix=f"{API_PREFIX}/gen",             tags=["Generative AI"])
app.include_router(pdf_router,     prefix=f"{API_PREFIX}/gen",             tags=["Generative AI"])
app.include_router(medical_router, prefix=f"{API_PREFIX}/gen",             tags=["Generative AI"])
app.include_router(vision_router,  prefix=f"{API_PREFIX}/vision",          tags=["Computer Vision"])
app.include_router(market_router,  prefix=f"{API_PREFIX}/agentic",         tags=["Agentic"])


@app.get(f"{API_PREFIX}/health", response_model=HealthStatus, tags=["Platform"])
async def health():
    """Platform health check — reports load status per module."""
    from pathlib import Path
    from core.config import (
        FLIGHT_FARE_MODEL, HOUSE_PRICE_MODEL, GOLD_PRICE_MODEL,
        HEART_DISEASE_MODEL, SPAM_MODEL, CHEST_DISEASE_MODEL, GEMINI_API_KEY,
    )
    return HealthStatus(
        status="ok",
        modules={
            "flight_fare":    FLIGHT_FARE_MODEL.exists(),
            "house_prices":   HOUSE_PRICE_MODEL.exists(),
            "gold_prices":    GOLD_PRICE_MODEL.exists(),
            "heart_disease":  HEART_DISEASE_MODEL.exists(),
            "spam_filter":    SPAM_MODEL.exists(),
            "chest_disease":  CHEST_DISEASE_MODEL.exists(),
            "generative_ai":  bool(GEMINI_API_KEY),
            "computer_vision": True,   # no weights needed, uses MediaPipe bundled models
            "agentic":        bool(GEMINI_API_KEY),
        },
    )
