from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

_rembg_session: Any | None = None


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


def create_transparent_portrait(source_path: Path, target_path: Path) -> dict[str, Any]:
    with Image.open(source_path) as source:
        source = source.convert("RGBA")
        segmented = _segment_with_grabcut(source) or _segment_with_rembg(source)
        if segmented is None:
            return {"ok": False, "message": "人像抠图失败，请使用纯色背景重新拍摄"}

        alpha = segmented.getchannel("A").filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.45))
        coverage = float(ImageStat.Stat(alpha).mean[0]) / 255
        if coverage < 0.10 or coverage > 0.88:
            return {"ok": False, "message": "人像边界识别失败，请换纯色背景重新拍摄", "coverage": coverage}

        segmented.putalpha(alpha)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        segmented.save(target_path, "PNG")
        return {"ok": True, "coverage": coverage}


def _segment_with_rembg(source: Image.Image) -> Image.Image | None:
    global _rembg_session
    try:
        from rembg import new_session, remove

        if _rembg_session is None:
            _rembg_session = new_session("u2netp")
        result = remove(
            source,
            session=_rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=8,
        )
        return result.convert("RGBA")
    except Exception:
        return None


def _segment_with_grabcut(source: Image.Image) -> Image.Image | None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    original_width, original_height = source.size
    max_side = 900
    scale = min(1.0, max_side / max(original_width, original_height))
    working = source.resize((int(original_width * scale), int(original_height * scale))) if scale < 1 else source
    rgb = working.convert("RGB")
    image = np.array(rgb)
    height, width = image.shape[:2]
    rect = (
        max(1, int(width * 0.04)),
        max(1, int(height * 0.02)),
        max(2, int(width * 0.92)),
        max(2, int(height * 0.94)),
    )
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 7, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.GaussianBlur(foreground, (5, 5), 0)

    result = source.copy()
    alpha = Image.fromarray(foreground, mode="L")
    if scale < 1:
        alpha = alpha.resize((original_width, original_height), Image.Resampling.LANCZOS)
    result.putalpha(alpha)
    return result
