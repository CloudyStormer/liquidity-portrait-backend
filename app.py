from __future__ import annotations

import hashlib
import shutil
import threading
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from auth import exchange_wechat_code, sign_auth_token, verify_auth_token
from photo_tools import create_transparent_portrait, validate_id_photo, warmup_portrait_matting
from store import ensure_data_dirs, read_store, update_store
from users import add_rewarded_bonus, get_usage, get_user, identify_user, list_user_history, update_user_profile
from utils import format_file_size, make_id, now_iso
from wechat_security import media_check_async

remove_methods = {"screenshot", "doodle", "selection"}
PHOTO_OUTPUT_SPECS = {
    "one-inch": {"width": 295, "height": 413, "topMarginRatio": 0.055, "personWidthRatio": 1.08, "maxPersonHeightRatio": 1.24},
    "two-inch": {"width": 413, "height": 579, "topMarginRatio": 0.055, "personWidthRatio": 1.08, "maxPersonHeightRatio": 1.24},
    "small-one-inch": {"width": 260, "height": 378, "topMarginRatio": 0.055, "personWidthRatio": 1.08, "maxPersonHeightRatio": 1.24},
    "large-one-inch": {"width": 390, "height": 567, "topMarginRatio": 0.055, "personWidthRatio": 1.08, "maxPersonHeightRatio": 1.24},
    "passport": {"width": 390, "height": 567, "topMarginRatio": 0.060, "personWidthRatio": 1.06, "maxPersonHeightRatio": 1.20},
    "social-security": {"width": 358, "height": 441, "topMarginRatio": 0.050, "personWidthRatio": 1.06, "maxPersonHeightRatio": 1.22},
}


class WechatUserInfo(BaseModel):
    nickName: str | None = None
    nickname: str | None = None
    avatarUrl: str | None = None


class WechatLoginBody(BaseModel):
    code: str = Field(min_length=1)
    clientId: str = Field(min_length=8)
    platform: str | None = None
    userInfo: WechatUserInfo | None = None


class IdentifyBody(BaseModel):
    clientId: str = Field(min_length=8)
    platform: str | None = None
    nickname: str | None = None
    avatarUrl: str | None = None
    openid: str | None = None


class PhotoUsageBody(BaseModel):
    id: str = Field(min_length=1)
    userId: str | None = None
    openid: str | None = None
    sourceType: Literal["album", "camera"] | None = None
    sizeId: str = Field(min_length=1)
    sizeName: str = Field(min_length=1)
    imagePath: str | None = None
    originalImagePath: str | None = None
    createdAt: str = Field(min_length=1)
    status: Literal["completed"]
    backgroundId: str | None = None
    backgroundColor: str | None = None


class ClientLogBody(BaseModel):
    event: str = Field(min_length=1, max_length=80)
    userId: str | None = None
    openid: str | None = None
    platform: str | None = None
    meta: dict[str, Any] | None = None


class RewardBody(BaseModel):
    userId: str = Field(min_length=1)
    placement: Literal["quota", "download"] = "quota"


class UserProfileBody(BaseModel):
    nickname: str | None = Field(default=None, max_length=80)
    avatarUrl: str | None = None


def create_app() -> FastAPI:
    ensure_data_dirs()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Liquidity Portrait Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")

    @app.on_event("startup")
    def warmup_photo_model() -> None:
        def run_warmup() -> None:
            try:
                warmup_portrait_matting()
            except Exception:
                pass

        threading.Thread(target=run_warmup, daemon=True).start()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "liquidity-portrait-backend", "runtime": "python", "time": now_iso()}

    @app.post("/api/auth/wechat/login")
    def wechat_login(body: WechatLoginBody) -> dict[str, Any]:
        wechat_session = exchange_wechat_code(body.code)
        nickname = None
        avatar_url = None
        if body.userInfo:
            nickname = body.userInfo.nickName or body.userInfo.nickname
            avatar_url = body.userInfo.avatarUrl

        user = identify_user(
            {
                "clientId": body.clientId,
                "platform": body.platform or "weapp",
                "nickname": nickname,
                "avatarUrl": avatar_url,
                "openid": wechat_session["openid"],
                "unionid": wechat_session.get("unionid"),
            }
        )
        token = sign_auth_token(user["id"], wechat_session["openid"] or "", user.get("platform") or "weapp")

        def mutate(store: dict[str, Any]) -> None:
            store["clientLogs"].append(
                {
                    "id": make_id("log"),
                    "event": "auth.wechat.login",
                    "userId": user["id"],
                    "openid": user.get("openid"),
                    "platform": user.get("platform"),
                    "meta": {"clientId": body.clientId, "hasAvatar": bool(avatar_url)},
                    "createdAt": now_iso(),
                }
            )

        update_store(mutate)

        return {
            "token": token,
            "user": {
                "id": user["id"],
                "platform": user.get("platform"),
                "nickname": user.get("nickname"),
                "avatarUrl": user.get("avatarUrl"),
                "openid": user.get("openid"),
                "unionid": user.get("unionid"),
                "openaiUserId": user.get("openaiUserId"),
                "createdAt": user.get("createdAt"),
                "lastSeenAt": user.get("lastSeenAt"),
            },
            "usage": get_usage(user["id"]),
        }

    @app.post("/api/users/identify")
    def identify(body: IdentifyBody) -> dict[str, Any]:
        user = identify_user(body.model_dump())
        return {
            "user": {
                "id": user["id"],
                "platform": user.get("platform"),
                "nickname": user.get("nickname"),
                "avatarUrl": user.get("avatarUrl"),
                "openid": user.get("openid"),
                "openaiUserId": user.get("openaiUserId"),
                "createdAt": user.get("createdAt"),
                "lastSeenAt": user.get("lastSeenAt"),
            },
            "usage": get_usage(user["id"]),
        }

    @app.patch("/api/users/{user_id}/profile")
    def update_profile(user_id: str, body: UserProfileBody, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_user_token(authorization, user_id)
        try:
            user = update_user_profile(user_id, body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="USER_NOT_FOUND") from exc
        return {"user": to_public_user(user)}

    @app.post("/api/users/{user_id}/avatar")
    async def upload_avatar(user_id: str, file: UploadFile = File(...), authorization: str | None = Header(default=None)) -> dict[str, Any]:
        payload = require_user_token(authorization, user_id)
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="USER_NOT_FOUND")
        suffix = Path(file.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        avatar_dir = config.UPLOAD_DIR / "avatars"
        avatar_dir.mkdir(parents=True, exist_ok=True)
        avatar_path = avatar_dir / f"{user_id}{suffix}"
        with avatar_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        avatar_url = public_url_for(avatar_path)
        security_result = media_check_async(avatar_url, str(payload.get("openid") or user.get("openid") or ""))
        updated_user = update_user_profile(user_id, {"avatarUrl": avatar_url})
        return {"user": to_public_user(updated_user), "security": {"traceId": security_result.get("trace_id")}}

    @app.post("/api/photo/usage-records")
    def photo_usage_records(body: PhotoUsageBody, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        record = body.model_dump()
        if not record.get("userId"):
            raise HTTPException(status_code=400, detail="USER_ID_REQUIRED")
        require_user_token(authorization, record["userId"])
        stored_record = {
            "id": record["id"],
            "userId": record["userId"],
            "sourceType": record.get("sourceType"),
            "sizeId": record["sizeId"],
            "sizeName": record["sizeName"],
            "createdAt": record["createdAt"],
            "status": record["status"],
        }

        def mutate(store: dict[str, Any]) -> None:
            records = store["photoUsageRecords"]
            index = next((idx for idx, item in enumerate(records) if item.get("id") == stored_record["id"]), -1)
            if index >= 0:
                records[index] = stored_record
            else:
                records.append(stored_record)

        update_store(mutate)
        return {"ok": True, "record": stored_record}

    @app.post("/api/photo/validate")
    async def validate_photo(
        image: UploadFile = File(...),
        userId: str = Form(...),
        sizeId: str = Form(...),
        sourceType: Literal["album", "camera"] = Form(...),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = require_user_token(authorization, userId)
        user = get_user(userId)
        if not user:
            raise HTTPException(status_code=404, detail="USER_NOT_FOUND")

        suffix = Path(image.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        original_path = config.UPLOAD_DIR / f"{make_id('photo')}{suffix}"
        with original_path.open("wb") as output:
            shutil.copyfileobj(image.file, output)

        original_url = public_url_for(original_path)
        wechat_result = media_check_async(original_url, str(payload.get("openid") or user.get("openid") or ""))
        validation = validate_id_photo(original_path)
        if not validation["ok"]:
            return {
                "ok": False,
                "message": validation["issues"][0],
                "validation": validation,
                "security": {"traceId": wechat_result.get("trace_id")},
            }

        processed_path = config.UPLOAD_DIR / "processed" / f"{original_path.stem}.png"
        try:
            cutout_result = create_transparent_portrait(original_path, processed_path, PHOTO_OUTPUT_SPECS.get(sizeId, PHOTO_OUTPUT_SPECS["one-inch"]))
        except Exception as exc:
            cutout_result = {"ok": False, "message": "人像抠图服务异常，请稍后重试", "error": exc.__class__.__name__}
        if not cutout_result.get("ok"):
            return {
                "ok": False,
                "message": cutout_result.get("message") or "人像抠图失败，请使用纯色背景重新拍摄",
                "validation": validation,
                "security": {"traceId": wechat_result.get("trace_id")},
                "cutout": cutout_result,
            }
        return {
            "ok": True,
            "message": "照片合规",
            "imagePath": public_url_for(processed_path),
            "originalUrl": original_url,
            "validation": validation,
            "security": {"traceId": wechat_result.get("trace_id")},
            "cutout": cutout_result,
            "meta": {"sizeId": sizeId, "sourceType": sourceType},
        }

    @app.post("/api/logs")
    def logs(body: ClientLogBody) -> dict[str, bool]:
        log = body.model_dump()

        def mutate(store: dict[str, Any]) -> None:
            store["clientLogs"].append({"id": make_id("log"), **log, "createdAt": now_iso()})

        update_store(mutate)
        return {"ok": True}

    @app.get("/api/users/{user_id}/usage")
    def usage(user_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_user_token(authorization, user_id)
        user = get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="USER_NOT_FOUND")
        return {"usage": get_usage(user["id"])}

    @app.get("/api/users/{user_id}/history")
    def history(user_id: str, type: str | None = None, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_user_token(authorization, user_id)
        record_type = type if type in {"image", "md5", "photo"} else None
        records = list_user_history(user_id, record_type)
        if record_type == "photo":
            return {"records": records}
        return {"records": [to_history_item(item) for item in records]}

    @app.post("/api/ads/reward")
    def reward(body: RewardBody) -> dict[str, Any]:
        if body.placement == "quota":
            try:
                usage_data = add_rewarded_bonus(body.userId)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="USER_NOT_FOUND") from exc
        else:
            usage_data = get_usage(body.userId)
        return {"ok": True, "usage": usage_data}

    @app.post("/api/process/image")
    async def process_image(
        image: UploadFile = File(...),
        userId: str = Form(...),
        method: str = Form("screenshot"),
        markerDataUrl: str | None = Form(None),
        rightsConfirmed: bool = Form(False),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = require_user_token(authorization, userId)
        if method not in remove_methods:
            raise HTTPException(status_code=400, detail="INVALID_METHOD")
        if not rightsConfirmed:
            raise HTTPException(status_code=400, detail="RIGHTS_CONFIRMATION_REQUIRED")
        user = get_user(userId)
        if not user:
            raise HTTPException(status_code=404, detail="USER_NOT_FOUND")
        usage_data = get_usage(user["id"])
        if usage_data["remaining"] <= 0:
            raise HTTPException(status_code=429, detail={"error": "QUOTA_EXHAUSTED", "usage": usage_data})

        suffix = Path(image.filename or "").suffix or ".jpg"
        original_path = config.UPLOAD_DIR / f"{make_id('upload')}{suffix}"
        with original_path.open("wb") as file:
            shutil.copyfileobj(image.file, file)
        original_url = public_url_for(original_path)
        security_result = media_check_async(original_url, str(payload.get("openid") or user.get("openid") or ""))

        request_id = f"oai_{make_id('req')}"
        record = {
            "id": make_id("pf"),
            "type": "image",
            "userId": user["id"],
            "originalName": image.filename or "image",
            "originalUrl": original_url,
            "processedUrl": original_url,
            "fileSize": original_path.stat().st_size,
            "method": method,
            "status": "completed",
            "provider": "local-preview",
            "openaiRequestId": request_id,
            "wechatSecurityTraceId": security_result.get("trace_id"),
            "createdAt": now_iso(),
        }
        request_record = {
            "id": request_id,
            "userId": user["id"],
            "openaiUserId": user.get("openaiUserId"),
            "endpoint": "images.edit",
            "model": config.OPENAI_IMAGE_MODEL,
            "status": "skipped",
            "createdAt": now_iso(),
        }

        def mutate(store: dict[str, Any]) -> None:
            store["history"].append(record)
            store["openaiRequests"].append(request_record)

        update_store(mutate)
        return {"file": to_processed_file(record), "usage": get_usage(user["id"])}

    @app.post("/api/tools/md5")
    async def md5_tool(file: UploadFile = File(...), userId: str = Form(...), authorization: str | None = Header(default=None)) -> dict[str, Any]:
        payload = require_user_token(authorization, userId)
        user = get_user(userId)
        if not user:
            raise HTTPException(status_code=404, detail="USER_NOT_FOUND")

        suffix = Path(file.filename or "").suffix.lower() or ".bin"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".mp3", ".wav", ".m4a"}:
            raise HTTPException(status_code=400, detail="UNSUPPORTED_UPLOAD_MEDIA_TYPE")
        upload_path = config.UPLOAD_DIR / "security" / f"{make_id('security')}{suffix}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        with upload_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        media_type = 1 if suffix in {".mp3", ".wav", ".m4a"} else 2
        security_result = media_check_async(public_url_for(upload_path), str(payload.get("openid") or user.get("openid") or ""), media_type=media_type)

        content = upload_path.read_bytes()
        md5 = hashlib.md5(content).hexdigest()
        store = read_store()
        duplicate = any(item.get("type") == "md5" and item.get("md5") == md5 for item in store["history"])
        record = {
            "id": make_id("md5"),
            "type": "md5",
            "userId": user["id"],
            "fileName": file.filename or "file",
            "fileSize": len(content),
            "md5": md5,
            "duplicate": duplicate,
            "wechatSecurityTraceId": security_result.get("trace_id"),
            "createdAt": now_iso(),
        }

        def mutate(next_store: dict[str, Any]) -> None:
            next_store["history"].append(record)

        update_store(mutate)
        return {"result": to_md5_result(record)}

    return app


def require_user_token(authorization: str | None, user_id: str) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="AUTH_REQUIRED")
    payload = verify_auth_token(authorization.removeprefix("Bearer ").strip())
    if payload.get("userId") != user_id:
        raise HTTPException(status_code=403, detail="USER_FORBIDDEN")
    return payload


def public_url_for(file_path: Path) -> str:
    try:
        relative = file_path.resolve().relative_to(config.UPLOAD_DIR.resolve()).as_posix()
    except ValueError:
        relative = file_path.name
    return f"{public_base_url()}/uploads/{relative}"


def public_base_url() -> str:
    base_url = (config.PUBLIC_BASE_URL or "").rstrip("/")
    if not base_url or "localhost" in base_url or "127.0.0.1" in base_url:
        return "https://api.hgshouse.com/portrait"
    return base_url


def to_public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user["id"],
        "platform": user.get("platform"),
        "nickname": user.get("nickname"),
        "avatarUrl": user.get("avatarUrl"),
        "openid": user.get("openid"),
        "unionid": user.get("unionid"),
        "openaiUserId": user.get("openaiUserId"),
        "createdAt": user.get("createdAt"),
        "lastSeenAt": user.get("lastSeenAt"),
    }


def to_processed_file(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "type": "image",
        "originalName": record.get("originalName"),
        "thumb": record.get("originalUrl"),
        "processedUrl": record.get("processedUrl"),
        "processTime": record.get("createdAt"),
        "fileSize": format_file_size(int(record.get("fileSize") or 0)),
        "method": record.get("method"),
        "provider": record.get("provider"),
        "openaiRequestId": record.get("openaiRequestId"),
    }


def to_md5_result(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "fileName": record.get("fileName"),
        "fileSize": format_file_size(int(record.get("fileSize") or 0)),
        "md5": record.get("md5"),
        "calcTime": record.get("createdAt"),
        "duplicate": record.get("duplicate"),
    }


def to_history_item(record: dict[str, Any]) -> dict[str, Any]:
    return to_processed_file(record) if record.get("type") == "image" else to_md5_result(record)


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=config.PORT, reload=config.UVICORN_RELOAD)
