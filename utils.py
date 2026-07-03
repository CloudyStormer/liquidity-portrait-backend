from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:18]}"


def openai_user_id(user_id: str) -> str:
    return f"lp_{hashlib.sha256(user_id.encode('utf-8')).hexdigest()[:32]}"


def format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"
