from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

from fastapi import HTTPException

import config

_token_cache: dict[str, Any] = {"token": "", "expires_at": 0}


def get_wechat_access_token() -> str:
    if not config.WECHAT_APP_ID or not config.WECHAT_APP_SECRET:
        if config.WECHAT_DEV_OPENID:
            return ""
        raise HTTPException(status_code=500, detail="WECHAT_CONFIG_MISSING")

    now = int(time.time())
    if _token_cache["token"] and int(_token_cache["expires_at"]) - 120 > now:
        return str(_token_cache["token"])

    query = urllib.parse.urlencode(
        {
            "grant_type": "client_credential",
            "appid": config.WECHAT_APP_ID,
            "secret": config.WECHAT_APP_SECRET,
        }
    )
    url = f"https://api.weixin.qq.com/cgi-bin/token?{query}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="WECHAT_ACCESS_TOKEN_FAILED") from exc

    if data.get("errcode") or not data.get("access_token"):
        raise HTTPException(status_code=502, detail=data.get("errmsg") or "WECHAT_ACCESS_TOKEN_FAILED")

    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + int(data.get("expires_in") or 7200)
    return str(_token_cache["token"])


def media_check_async(media_url: str, openid: str, scene: int = 1) -> dict[str, Any]:
    token = get_wechat_access_token()
    if not token:
        return {"errcode": 0, "errmsg": "dev bypass"}

    url = f"https://api.weixin.qq.com/wxa/media_check_async?access_token={urllib.parse.quote(token)}"
    payload = json.dumps(
        {
            "media_url": media_url,
            "media_type": 2,
            "version": 2,
            "openid": openid,
            "scene": scene,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="WECHAT_MEDIA_CHECK_FAILED") from exc

    if data.get("errcode") not in (0, None):
        raise HTTPException(status_code=400, detail={"error": "WECHAT_MEDIA_CHECK_REJECTED", "wechat": data})
    return data
