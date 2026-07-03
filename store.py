from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from config import DATA_DIR

STORE_PATH = DATA_DIR / "store.json"
_lock = threading.Lock()


def empty_store() -> dict[str, Any]:
    return {
        "users": [],
        "history": [],
        "photoUsageRecords": [],
        "clientLogs": [],
        "openaiRequests": [],
    }


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_store() -> dict[str, Any]:
    ensure_data_dirs()
    if not STORE_PATH.exists():
        return empty_store()
    with STORE_PATH.open("r", encoding="utf-8-sig") as file:
        parsed = json.load(file)

    store = empty_store()
    for key in store:
        value = parsed.get(key)
        store[key] = value if isinstance(value, list) else []
    return store


def write_store(store: dict[str, Any]) -> None:
    ensure_data_dirs()
    temp_path = Path(f"{STORE_PATH}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(store, file, ensure_ascii=False, indent=2)
    temp_path.replace(STORE_PATH)


def update_store(mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    with _lock:
        store = read_store()
        mutator(store)
        write_store(store)
        return store
