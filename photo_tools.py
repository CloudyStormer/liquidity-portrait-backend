from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat


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


def create_transparent_portrait(source_path: Path, target_path: Path) -> None:
    with Image.open(source_path) as image:
        image = image.convert("RGBA")
        width, height = image.size
        rgb = image.convert("RGB")

        sample_points = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width // 2, 0),
            (width // 2, height - 1),
        ]
        colors = [rgb.getpixel(point) for point in sample_points]
        bg = tuple(int(sum(color[index] for color in colors) / len(colors)) for index in range(3))

        mask = Image.new("L", (width, height), 0)
        pixels = rgb.load()
        mask_pixels = mask.load()
        center_x = width / 2
        center_y = height * 0.46
        radius_x = width * 0.36
        radius_y = height * 0.43

        for y in range(height):
            for x in range(width):
                pixel = pixels[x, y]
                distance = math.sqrt(sum((pixel[index] - bg[index]) ** 2 for index in range(3)))
                center_score = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2
                keep = distance > 34 or center_score < 0.78
                mask_pixels[x, y] = 255 if keep else 0

        mask = mask.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(1.2))
        coverage = float(ImageStat.Stat(mask).mean[0]) / 255
        if coverage < 0.18:
            mask = Image.new("L", (width, height), 255)
        image.putalpha(mask)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(target_path, "PNG")
