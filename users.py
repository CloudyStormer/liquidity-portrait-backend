from __future__ import annotations

from typing import Any

import config
from store import read_store, update_store
from utils import make_id, now_iso, openai_user_id, today_key


def identify_user(input_data: dict[str, Any]) -> dict[str, Any]:
    now = now_iso()
    result: dict[str, Any] = {}

    def mutate(store: dict[str, Any]) -> None:
        nonlocal result
        users = store["users"]
        user = None
        openid = input_data.get("openid")
        client_id = input_data["clientId"]

        if openid:
            user = next((item for item in users if item.get("openid") == openid), None)
        if not user:
            user = next((item for item in users if item.get("clientId") == client_id), None)

        if not user:
            user_id = make_id("usr")
            user = {
                "id": user_id,
                "clientId": client_id,
                "platform": input_data.get("platform") or "web",
                "nickname": input_data.get("nickname"),
                "avatarUrl": input_data.get("avatarUrl"),
                "openid": openid,
                "unionid": input_data.get("unionid"),
                "openaiUserId": openai_user_id(user_id),
                "createdAt": now,
                "lastSeenAt": now,
                "quotaBonuses": {},
            }
            users.append(user)
        else:
            user["platform"] = input_data.get("platform") or user.get("platform")
            user["nickname"] = input_data.get("nickname") or user.get("nickname")
            user["avatarUrl"] = input_data.get("avatarUrl") or user.get("avatarUrl")
            user["openid"] = openid or user.get("openid")
            user["unionid"] = input_data.get("unionid") or user.get("unionid")
            user["lastSeenAt"] = now

        result = user

    update_store(mutate)
    return result


def get_user(user_id: str) -> dict[str, Any] | None:
    store = read_store()
    return next((item for item in store["users"] if item.get("id") == user_id), None)


def update_user_profile(user_id: str, input_data: dict[str, Any]) -> dict[str, Any]:
    now = now_iso()
    result: dict[str, Any] = {}

    def mutate(store: dict[str, Any]) -> None:
        nonlocal result
        user = next((item for item in store["users"] if item.get("id") == user_id), None)
        if not user:
            raise ValueError("USER_NOT_FOUND")
        if input_data.get("nickname"):
            user["nickname"] = input_data["nickname"]
        if input_data.get("avatarUrl"):
            user["avatarUrl"] = input_data["avatarUrl"]
        user["lastSeenAt"] = now
        result = user

    update_store(mutate)
    return result


def get_usage(user_id: str) -> dict[str, Any]:
    store = read_store()
    date = today_key()
    user = next((item for item in store["users"] if item.get("id") == user_id), None)
    used = len(
        [
            item
            for item in store["history"]
            if item.get("userId") == user_id and item.get("type") == "image" and str(item.get("createdAt", "")).startswith(date)
        ]
    )
    bonus = (user or {}).get("quotaBonuses", {}).get(date, 0)
    total = config.FREE_DAILY_QUOTA + bonus
    return {
        "date": date,
        "used": used,
        "total": total,
        "remaining": max(0, total - used),
        "freeDailyQuota": config.FREE_DAILY_QUOTA,
        "bonus": bonus,
    }


def add_rewarded_bonus(user_id: str) -> dict[str, Any]:
    date = today_key()

    def mutate(store: dict[str, Any]) -> None:
        user = next((item for item in store["users"] if item.get("id") == user_id), None)
        if not user:
            raise ValueError("USER_NOT_FOUND")
        user.setdefault("quotaBonuses", {})
        user["quotaBonuses"][date] = user["quotaBonuses"].get(date, 0) + config.REWARDED_AD_BONUS
        user["lastSeenAt"] = now_iso()

    update_store(mutate)
    return get_usage(user_id)


def list_user_history(user_id: str, record_type: str | None = None) -> list[dict[str, Any]]:
    store = read_store()
    if record_type == "photo":
        return list_photo_usage_records(user_id)
    records = [
        item
        for item in store["history"]
        if item.get("userId") == user_id and (not record_type or item.get("type") == record_type)
    ]
    return sorted(records, key=lambda item: str(item.get("createdAt", "")), reverse=True)


def list_photo_usage_records(user_id: str) -> list[dict[str, Any]]:
    store = read_store()
    records = [
        item
        for item in store["photoUsageRecords"]
        if item.get("userId") == user_id
    ]
    return sorted(records, key=lambda item: str(item.get("createdAt", "")), reverse=True)
