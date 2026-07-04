from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

_rembg_sessions: dict[str, Any] = {}


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
        segmented, model_name, errors = _segment_with_rembg(source)
        if segmented is None:
            return {"ok": False, "message": "人像抠图失败，请使用纯色背景重新拍摄", "errors": errors}

        if output_size:
            segmented = _compose_to_id_canvas(segmented, output_size)

        alpha = _refine_alpha(segmented.getchannel("A"))
        coverage = float(ImageStat.Stat(alpha).mean[0]) / 255
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
        }


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
            result = remove(
                source,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=230,
                alpha_matting_background_threshold=18,
                alpha_matting_erode_size=2,
                post_process_mask=True,
            )
            return result.convert("RGBA"), model_name
        except Exception as exc:
            errors.append(f"{model_name} alpha matting failed: {exc.__class__.__name__}: {exc}")
            try:
                session = _rembg_sessions.get(model_name)
                if session is None:
                    session = new_session(model_name)
                    _rembg_sessions[model_name] = session
                result = remove(source, session=session, post_process_mask=True)
                return result.convert("RGBA"), model_name, errors
            except Exception as fallback_exc:
                errors.append(f"{model_name} fallback failed: {fallback_exc.__class__.__name__}: {fallback_exc}")
                continue

    return None, None, errors


def _matting_models() -> list[str]:
    configured = os.getenv("PORTRAIT_MATTING_MODELS", "")
    models = [item.strip() for item in configured.split(",") if item.strip()]
    return models or ["birefnet-portrait", "u2net_human_seg", "isnet-general-use", "u2net"]


def _refine_alpha(alpha: Image.Image) -> Image.Image:
    return alpha.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.35))


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
