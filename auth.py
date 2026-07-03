from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from typing import Any

from fastapi import HTTPException

import config


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def sign_auth_token(user_id: str, openid: str, platform: str) -> str:
    payload = {
        "userId": user_id,
        "openid": openid,
        "platform": platform,
        "exp": int(time.time()) + config.AUTH_TOKEN_TTL_SECONDS,
    }
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(config.AUTH_TOKEN_SECRET.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64url(signature)}"


def verify_auth_token(token: str) -> dict[str, Any]:
    try:
        body, signature = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="INVALID_AUTH_TOKEN") from exc

    expected = hmac.new(config.AUTH_TOKEN_SECRET.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url(expected), signature):
        raise HTTPException(status_code=401, detail="INVALID_AUTH_TOKEN")

    payload = json.loads(_b64url_decode(body).decode("utf-8"))
    if not payload.get("userId") or not payload.get("openid") or int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="INVALID_AUTH_TOKEN")
    return payload


def exchange_wechat_code(code: str) -> dict[str, str | None]:
    if not config.WECHAT_APP_ID or not config.WECHAT_APP_SECRET:
        if config.WECHAT_DEV_OPENID:
            return {"openid": config.WECHAT_DEV_OPENID, "unionid": config.WECHAT_DEV_UNIONID or None}
        raise HTTPException(status_code=500, detail="WECHAT_CONFIG_MISSING")

    query = urllib.parse.urlencode(
        {
            "appid": config.WECHAT_APP_ID,
            "secret": config.WECHAT_APP_SECRET,
            "js_code": code,
            "grant_type": "authorization_code",
        }
    )
    url = f"https://api.weixin.qq.com/sns/jscode2session?{query}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="WECHAT_CODE_EXCHANGE_FAILED") from exc

    if data.get("errcode") or not data.get("openid"):
        raise HTTPException(status_code=401, detail=data.get("errmsg") or "WECHAT_CODE_EXCHANGE_FAILED")
    return {"openid": data["openid"], "unionid": data.get("unionid")}
