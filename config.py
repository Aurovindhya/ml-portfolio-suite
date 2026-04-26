"""
core/config.py

Centralised settings loaded from environment variables.
All modules import from here — no scattered os.getenv() calls.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# ── Google Gemini ─────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Langfuse ─────────────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── Model weights paths ───────────────────────────────────────────────────────
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

FLIGHT_FARE_MODEL    = WEIGHTS_DIR / "flight_fare.pkl"
HOUSE_PRICE_MODEL    = WEIGHTS_DIR / "house_price.pkl"
GOLD_PRICE_MODEL     = WEIGHTS_DIR / "gold_price.pkl"
HEART_DISEASE_MODEL  = WEIGHTS_DIR / "heart_disease.pkl"
SPAM_MODEL           = WEIGHTS_DIR / "spam.pkl"
CHEST_DISEASE_MODEL  = WEIGHTS_DIR / "chest_disease.pth"

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── API ───────────────────────────────────────────────────────────────────────
API_VERSION = "v1"
API_PREFIX  = f"/api/{API_VERSION}"
