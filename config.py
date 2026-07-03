from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = ROOT_DIR / "uploads"


def int_from_env(name: str, fallback: int) -> int:
    try:
        value = int(os.getenv(name, ""))
        return value if value > 0 else fallback
    except ValueError:
        return fallback


PORT = int_from_env("PORT", 8787)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8787")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

AUTH_TOKEN_SECRET = os.getenv("AUTH_TOKEN_SECRET", "liquidity-portrait-dev-secret")
AUTH_TOKEN_TTL_SECONDS = int_from_env("AUTH_TOKEN_TTL_SECONDS", 7 * 24 * 60 * 60)

WECHAT_APP_ID = os.getenv("WECHAT_APP_ID", "")
WECHAT_APP_SECRET = os.getenv("WECHAT_APP_SECRET", "")
WECHAT_DEV_OPENID = os.getenv("WECHAT_DEV_OPENID", "")
WECHAT_DEV_UNIONID = os.getenv("WECHAT_DEV_UNIONID", "")

FREE_DAILY_QUOTA = int_from_env("FREE_DAILY_QUOTA", 3)
REWARDED_AD_BONUS = int_from_env("REWARDED_AD_BONUS", 3)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
ENABLE_OPENAI_IMAGE_EDIT = os.getenv("ENABLE_OPENAI_IMAGE_EDIT", "").lower() == "true"
