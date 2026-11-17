"""Citra sebagai data: ekstrak fitur visual ke tabel sehingga bisa di-cleaning, EDA, dan diagram."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .store import Dataset

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
MAX_PIXELS = 50_000_000
THUMB = 160
KEEP = 1024  # sisi terpanjang untuk salinan yang dikirim ke AI


def _sharpness(gray: np.ndarray) -> float:
    """Varians Laplacian: kecil = buram."""
    if min(gray.shape) < 3:
        return 0.0
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def _dominant_hex(img: Image.Image) -> str:
    small = img.convert("RGB").resize((64, 64))
    q = small.quantize(colors=5, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[:15]
    counts = sorted(q.getcolors(), reverse=True)
    idx = counts[0][1]
    r, g, b = pal[idx * 3: idx * 3 + 3]
    return f"#{r:02x}{g:02x}{b:02x}"


def image_features(name: str, data: bytes) -> tuple[dict, bytes, bytes]:
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    img = Image.open(io.BytesIO(data))
    fmt, mode = img.format, img.mode
    img = ImageOps.exif_transpose(img)
    img.load()
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    w, h = img.size
    gs = bool(np.abs(arr[..., 0] - arr[..., 1]).mean() < 2 and np.abs(arr[..., 1] - arr[..., 2]).mean() < 2)
    row = {
        "filename": name,
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 3) if h else None,
        "megapixels": round(w * h / 1e6, 3),
        "format": fmt,
        "mode": mode,
        "size_kb": round(len(data) / 1024, 1),
        "mean_r": round(float(arr[..., 0].mean()), 1),
        "mean_g": round(float(arr[..., 1].mean()), 1),
        "mean_b": round(float(arr[..., 2].mean()), 1),
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
        "sharpness": round(_sharpness(gray), 1),
        "is_grayscale": gs,
        "dominant_color": _dominant_hex(rgb),
    }
    t = rgb.copy()
    t.thumbnail((THUMB, THUMB))
    tb = io.BytesIO()
    t.save(tb, "JPEG", quality=80)
    k = rgb.copy()
    k.thumbnail((KEEP, KEEP))
    kb = io.BytesIO()
    k.save(kb, "JPEG", quality=85)
    return row, tb.getvalue(), kb.getvalue()


def images_to_dataset(name: str, files: list[tuple[str, bytes]]) -> Dataset:
    rows, thumbs, keeps, errors = [], {}, {}, []
    for fname, data in files:
        try:
            row, tb, kb = image_features(fname, data)
        except Exception as e:  # file rusak
            errors.append(f"{fname}: {e}")
            continue
        base, i = fname, 1
        while base in thumbs:
            i += 1
            base = f"{fname}#{i}"
        row["filename"] = base
        rows.append(row)
        thumbs[base] = tb
        keeps[base] = kb
    if not rows:
        raise ValueError("Gambar tidak dapat dibaca: " + "; ".join(errors[:3]))
    df = pd.DataFrame(rows)
    note = "Setiap baris = satu gambar; kolom adalah fitur visual (ukuran, kecerahan, ketajaman, warna dominan)."
    if errors:
        note += f" {len(errors)} file dilewati karena rusak."
    return Dataset(name, df, kind="image", meta={"source": "image", "thumbs": thumbs, "images": keeps, "note": note})
