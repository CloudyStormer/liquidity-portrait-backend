from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image, ImageFilter, ImageStat

_rembg_sessions: dict[str, Any] = {}
_onnx_sessions: dict[str, Any] = {}
_WARMUP_DONE = False
_MATTING_MAX_SIDE = int(os.getenv("PORTRAIT_MATTING_MAX_SIDE", "1280"))

_ONNX_MODEL_SPECS = {
    "u2net_human_seg": {
        "filename": "u2net_human_seg.onnx",
        "md5": "c09ddc2e0104f800e3e1bb4652583d1f",
        "urls": [
            os.getenv("PORTRAIT_U2NET_HUMAN_SEG_URL", ""),
            "https://ghproxy.net/https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
        ],
    },
    "u2netp": {
        "filename": "u2netp.onnx",
        "md5": "8e83ca70e441ab06c318d82300c84806",
        "urls": [
            os.getenv("PORTRAIT_U2NETP_URL", ""),
            "https://hf-mirror.com/BritishWerewolf/U-2-Netp/resolve/main/onnx/model.onnx",
            "https://ghproxy.net/https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        ],
    },
}


def validate_id_photo(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        ratio = width / height if height else 0
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = float(stat.mean[0])
        extrema = gray.getextrema()
        contrast = float(extrema[1] - extrema[0])

    issues: list[str] = []
    if width < 260 or height < 360:
        issues.append("照片像素过低，请选择更清晰的正面照片")
    if width >= height or ratio < 0.45 or ratio > 0.9:
        issues.append("照片比例不符合证件照要求，请上传竖版半身照片")
    if brightness < 45:
        issues.append("照片过暗，请在光线充足处重新拍摄")
    if brightness > 235:
        issues.append("照片过亮，请避免强光或过曝")
    if contrast < 24:
        issues.append("照片对比度过低，请更换清晰背景重新拍摄")

    return {
        "ok": not issues,
        "issues": issues,
        "width": width,
        "height": height,
        "ratio": ratio,
        "brightness": brightness,
        "contrast": contrast,
    }


def create_transparent_portrait(source_path: Path, target_path: Path, output_size: tuple[int, int] | None = None) -> dict[str, Any]:
    with Image.open(source_path) as source:
        source = source.convert("RGBA")
        original_size = source.size
        source = _resize_for_matting(source)
        segmented, model_name, errors = _segment_portrait(source)
        if segmented is None:
            return {"ok": False, "message": "人像抠图失败，请使用纯色背景重新拍摄", "errors": errors}

        alpha = _refine_alpha(segmented.getchannel("A"), source)
        source_coverage = _alpha_coverage(alpha)
        if source_coverage < 0.08 or source_coverage > 0.88:
            return {"ok": False, "message": "人像边界识别失败，请换纯色背景重新拍摄", "coverage": source_coverage, "model": model_name}
        segmented.putalpha(alpha)

        if output_size:
            segmented = _compose_to_id_canvas(segmented, output_size)

        alpha = _finalize_alpha(segmented.getchannel("A"))
        coverage = _alpha_coverage(alpha)
        if coverage < 0.08 or coverage > 0.88:
            return {"ok": False, "message": "人像边界识别失败，请换纯色背景重新拍摄", "coverage": coverage, "model": model_name}

        segmented.putalpha(alpha)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        segmented.save(target_path, "PNG")
        return {
            "ok": True,
            "coverage": coverage,
            "model": model_name,
            "width": segmented.width,
            "height": segmented.height,
            "sourceWidth": original_size[0],
            "sourceHeight": original_size[1],
            "workingWidth": source.width,
            "workingHeight": source.height,
        }


def warmup_portrait_matting() -> dict[str, Any]:
    global _WARMUP_DONE
    if _WARMUP_DONE:
        return {"ok": True, "warmed": True}
    canvas = Image.new("RGBA", (480, 640), (255, 255, 255, 255))
    result, model_name, errors = _segment_portrait(canvas)
    _WARMUP_DONE = result is not None
    return {"ok": _WARMUP_DONE, "model": model_name, "errors": errors}


def _segment_portrait(source: Image.Image) -> tuple[Image.Image | None, str | None, list[str]]:
    segmented, model_name, errors = _segment_with_onnx(source)
    if segmented is not None:
        return segmented, model_name, errors

    rembg_segmented, rembg_model_name, rembg_errors = _segment_with_rembg(source)
    return rembg_segmented, rembg_model_name, errors + rembg_errors


def _resize_for_matting(source: Image.Image) -> Image.Image:
    max_side = max(640, _MATTING_MAX_SIDE)
    if max(source.size) <= max_side:
        return source
    scale = max_side / max(source.size)
    target_size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    return source.resize(target_size, Image.Resampling.LANCZOS)


def _segment_with_onnx(source: Image.Image) -> tuple[Image.Image | None, str | None, list[str]]:
    errors: list[str] = []
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:
        return None, None, [f"onnxruntime import failed: {exc.__class__.__name__}: {exc}"]

    for model_name in _matting_models():
        if model_name not in _ONNX_MODEL_SPECS:
            continue
        try:
            model_path = _ensure_onnx_model(model_name)
            session_key = str(model_path)
            session = _onnx_sessions.get(session_key)
            if session is None:
                session_options = ort.SessionOptions()
                session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"])
                _onnx_sessions[session_key] = session

            rgb = source.convert("RGB")
            working = rgb.resize((320, 320), Image.Resampling.LANCZOS)
            image_array = np.asarray(working).astype(np.float32)
            image_array = image_array / max(float(image_array.max()), 1e-6)
            mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
            std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
            image_array = (image_array - mean) / std
            image_array = image_array.transpose((2, 0, 1))[None, :, :, :].astype(np.float32)
            prediction = session.run(None, {session.get_inputs()[0].name: image_array})[0][:, 0, :, :]
            minimum = float(prediction.min())
            maximum = float(prediction.max())
            prediction = (prediction - minimum) / max(maximum - minimum, 1e-6)
            mask = Image.fromarray((np.squeeze(prediction).clip(0, 1) * 255).astype("uint8"))
            mask = mask.resize(source.size, Image.Resampling.LANCZOS)
            result = source.copy()
            result.putalpha(mask)
            return result, model_name, errors
        except Exception as exc:
            errors.append(f"{model_name} onnx failed: {exc.__class__.__name__}: {exc}")

    return None, None, errors


def _segment_with_rembg(source: Image.Image) -> tuple[Image.Image | None, str | None, list[str]]:
    errors: list[str] = []
    try:
        from rembg import new_session, remove
    except Exception as exc:
        return None, None, [f"rembg import failed: {exc.__class__.__name__}: {exc}"]

    for model_name in _matting_models():
        try:
            session = _rembg_sessions.get(model_name)
            if session is None:
                session = new_session(model_name)
                _rembg_sessions[model_name] = session
            result = remove(source, session=session, post_process_mask=True)
            return result.convert("RGBA"), model_name, errors
        except Exception as exc:
            errors.append(f"{model_name} mask failed: {exc.__class__.__name__}: {exc}")
            try:
                session = _rembg_sessions.get(model_name)
                if session is None:
                    session = new_session(model_name)
                    _rembg_sessions[model_name] = session
                result = remove(
                    source,
                    session=session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=230,
                    alpha_matting_background_threshold=18,
                    alpha_matting_erode_size=2,
                    post_process_mask=True,
                )
                return result.convert("RGBA"), model_name, errors
            except Exception as fallback_exc:
                errors.append(f"{model_name} alpha matting failed: {fallback_exc.__class__.__name__}: {fallback_exc}")
                continue

    return None, None, errors


def _matting_models() -> list[str]:
    configured = os.getenv("PORTRAIT_MATTING_MODELS", "")
    models = [item.strip() for item in configured.split(",") if item.strip()]
    return models or ["u2net_human_seg", "u2netp", "birefnet-portrait", "isnet-general-use", "u2net"]


def _ensure_onnx_model(model_name: str) -> Path:
    spec = _ONNX_MODEL_SPECS[model_name]
    model_dir = Path(os.getenv("U2NET_HOME", Path.home() / ".u2net")).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / str(spec["filename"])
    expected_md5 = str(spec["md5"])
    if model_path.exists() and _file_md5(model_path) == expected_md5:
        return model_path

    download_errors: list[str] = []
    for url in [item for item in spec["urls"] if item]:
        try:
            _download_file(url, model_path, expected_md5)
            return model_path
        except Exception as exc:
            download_errors.append(f"{url}: {exc.__class__.__name__}: {exc}")
    raise RuntimeError("; ".join(download_errors) or f"{model_name} model download failed")


def _download_file(url: str, target_path: Path, expected_md5: str) -> None:
    request = Request(url, headers={"User-Agent": "liquidity-portrait-backend/1.0"})
    with urlopen(request, timeout=90) as response:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(target_path.parent), suffix=".tmp") as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(response, temp_file)
    try:
        if _file_md5(temp_path) != expected_md5:
            temp_path.unlink(missing_ok=True)
            raise ValueError("downloaded model checksum mismatch")
        temp_path.replace(target_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _file_md5(file_path: Path) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refine_alpha(alpha: Image.Image, source: Image.Image | None = None) -> Image.Image:
    alpha = _keep_primary_portrait_region(alpha)
    if source is not None:
        alpha = _apply_portrait_shape_prior(alpha, source)
    alpha = alpha.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.45))
    return alpha.point(_alpha_contrast_curve)


def _finalize_alpha(alpha: Image.Image) -> Image.Image:
    return alpha.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.25)).point(_alpha_contrast_curve)


def _alpha_coverage(alpha: Image.Image) -> float:
    return float(ImageStat.Stat(alpha).mean[0]) / 255


def _alpha_contrast_curve(value: int) -> int:
    if value <= 4:
        return 0
    if value >= 250:
        return 255
    return value


def _keep_primary_portrait_region(alpha: Image.Image) -> Image.Image:
    try:
        import numpy as np
    except Exception:
        return alpha

    original_size = alpha.size
    max_side = 560
    scale = min(1.0, max_side / max(original_size))
    working_alpha = alpha.resize(
        (max(1, round(original_size[0] * scale)), max(1, round(original_size[1] * scale))),
        Image.Resampling.BILINEAR,
    ) if scale < 1 else alpha

    alpha_array = np.asarray(working_alpha, dtype=np.uint8)
    hard_mask = alpha_array > 18
    height, width = alpha_array.shape
    visited = np.zeros((height, width), dtype=bool)
    center_x = width / 2
    best_pixels: list[tuple[int, int]] = []
    best_score = 0.0

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not hard_mask[start_y, start_x]:
                continue

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            pixels: list[tuple[int, int]] = []
            min_x = max_x = start_x
            max_y = start_y

            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and not visited[next_y, next_x]
                        and hard_mask[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

            area = len(pixels)
            if area < max(24, int(width * height * 0.0008)):
                continue

            x_center = (min_x + max_x + 1) / 2
            center_weight = 1.0 - min(0.65, abs(x_center - center_x) / max(width, 1))
            vertical_weight = 1.0 + ((max_y + 1) / max(height, 1)) * 0.18
            score = area * center_weight * vertical_weight
            if score > best_score:
                best_score = score
                best_pixels = pixels

    if not best_pixels:
        return alpha

    primary = np.zeros((height, width), dtype=np.uint8)
    for y, x in best_pixels:
        primary[y, x] = 255

    keep_mask = Image.fromarray(primary)
    if keep_mask.size != original_size:
        keep_mask = keep_mask.resize(original_size, Image.Resampling.NEAREST)
    keep_mask = keep_mask.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(2.0))
    alpha_array = np.asarray(alpha, dtype=np.uint8)
    keep_array = np.asarray(keep_mask, dtype=np.uint16)
    refined = (alpha_array.astype(np.uint16) * keep_array // 255).astype(np.uint8)
    return Image.fromarray(refined)


def _apply_portrait_shape_prior(alpha: Image.Image, source: Image.Image) -> Image.Image:
    try:
        import numpy as np
    except Exception:
        return alpha

    alpha_array = np.asarray(alpha, dtype=np.uint8)
    max_side = 560
    scale = min(1.0, max_side / max(alpha.size))
    analysis_size = (
        max(1, round(alpha.width * scale)),
        max(1, round(alpha.height * scale)),
    )
    analysis_alpha = alpha.resize(analysis_size, Image.Resampling.BILINEAR) if scale < 1 else alpha
    analysis_source = source.convert("RGB").resize(analysis_size, Image.Resampling.BILINEAR)

    rgb = np.asarray(analysis_source, dtype=np.uint8)
    analysis_alpha_array = np.asarray(analysis_alpha, dtype=np.uint8)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    skin = (
        (analysis_alpha_array > 80)
        & (red > 70)
        & (green > 45)
        & (blue > 30)
        & (red > green)
        & (green > blue)
        & ((red - blue) > 18)
        & (green * 100 > red * 48)
        & (blue * 100 > red * 30)
    )

    face_bbox = _largest_mask_bbox(skin)
    if face_bbox is None:
        return alpha

    left, top, right, bottom = face_bbox
    scale_x = alpha.width / analysis_alpha.width
    scale_y = alpha.height / analysis_alpha.height
    left = round(left * scale_x)
    right = round(right * scale_x)
    top = round(top * scale_y)
    bottom = round(bottom * scale_y)
    face_width = right - left + 1
    face_height = min(bottom - top + 1, int(face_width * 1.18))
    bottom = min(bottom, top + face_height)
    if face_width < alpha.width * 0.12 or face_height < alpha.height * 0.12:
        return alpha

    center_x = (left + right + 1) / 2
    y_indices, x_indices = np.indices(alpha_array.shape)
    head_top = max(0, top - int(face_height * 0.45))
    shoulder_start = bottom + face_height * 0.12
    shoulder_full = bottom + face_height * 0.85
    base_half_width = max(face_width * 0.46, alpha.width * 0.18)
    lower_half_width = max(face_width * 0.62, alpha.width * 0.24)
    progress = np.clip((y_indices - shoulder_start) / max(shoulder_full - shoulder_start, 1), 0, 1)
    allowed_half_width = base_half_width + (lower_half_width - base_half_width) * progress
    allowed_half_width = np.where(y_indices < head_top, face_width * 0.82, allowed_half_width)
    allowed = np.abs(x_indices - center_x) <= allowed_half_width
    protected_head = (y_indices <= bottom + face_height * 0.10) & (np.abs(x_indices - center_x) <= face_width * 0.50)
    allowed = allowed | protected_head

    shape_mask = Image.fromarray((allowed.astype(np.uint8) * 255))
    shape_mask = shape_mask.filter(ImageFilter.GaussianBlur(5.0))
    shape_array = np.asarray(shape_mask, dtype=np.uint16)
    refined = (alpha_array.astype(np.uint16) * shape_array // 255).astype(np.uint8)
    return Image.fromarray(refined)


def _largest_mask_bbox(mask: Any) -> tuple[int, int, int, int] | None:
    try:
        import numpy as np
    except Exception:
        return None

    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    best_area = 0
    best_bbox: tuple[int, int, int, int] | None = None

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not mask[start_y, start_x]:
                continue

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            area = 0
            min_x = max_x = start_x
            min_y = max_y = start_y
            while stack:
                y, x = stack.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and not visited[next_y, next_x]
                        and mask[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

            if area > best_area:
                best_area = area
                best_bbox = (min_x, min_y, max_x, max_y)

    return best_bbox


def _compose_to_id_canvas(segmented: Image.Image, output_size: tuple[int, int]) -> Image.Image:
    alpha = segmented.getchannel("A")
    bbox = alpha.point(lambda value: 255 if value > 8 else 0).getbbox()
    if not bbox:
        return segmented.resize(output_size, Image.Resampling.LANCZOS)

    left, top, right, bottom = bbox
    width, height = segmented.size
    person_width = right - left
    person_height = bottom - top
    pad_x = int(person_width * 0.10)
    pad_top = int(person_height * 0.05)
    pad_bottom = int(person_height * 0.12)
    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_top),
        min(width, right + pad_x),
        min(height, bottom + pad_bottom),
    )
    crop = segmented.crop(crop_box)

    target_width, target_height = output_size
    max_width = target_width * 0.94
    max_height = target_height * 0.94
    scale = min(max_width / crop.width, max_height / crop.height)
    resized_width = max(1, round(crop.width * scale))
    resized_height = max(1, round(crop.height * scale))
    crop = crop.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", output_size, (255, 255, 255, 0))
    x = round((target_width - resized_width) / 2)
    y = max(round(target_height * 0.035), round((target_height - resized_height) / 2))
    if y + resized_height > target_height:
        y = target_height - resized_height
    canvas.alpha_composite(crop, (x, y))
    return canvas
