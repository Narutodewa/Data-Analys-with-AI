"""Citra: deteksi gambar duplikat/mirip lewat perceptual hash (dHash), tanpa model besar."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image

from .common import need, result, table


def dhash(data: bytes, size: int = 8) -> int:
    im = Image.open(io.BytesIO(data)).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicates(ds, max_dist: int = 6) -> dict:
    imgs = ds.meta.get("images") or {}
    need(ds.kind == "image" and imgs, "Dataset ini bukan kumpulan gambar.")
    names = [n for n in ds.df["filename"] if n in imgs][:3000]
    need(len(names) >= 2, "Butuh minimal 2 gambar.")
    hs = [dhash(imgs[n]) for n in names]
    parent = list(range(len(names)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if bin(hs[i] ^ hs[j]).count("1") <= max_dist:
                parent[find(j)] = find(i)
    groups: dict[int, list[str]] = {}
    for i, n in enumerate(names):
        groups.setdefault(find(i), []).append(n)
    dup = [g for g in groups.values() if len(g) > 1]
    rows = [{"kelompok": f"G{k + 1}", "jumlah": len(g), "file": ", ".join(g[:6]) + (" …" if len(g) > 6 else "")} for k, g in enumerate(dup)]
    n_extra = sum(len(g) - 1 for g in dup)
    return result("Gambar duplikat / mirip", "perceptual hash (dHash 64-bit) + jarak Hamming", f"{len(dup)} kelompok gambar mirip; {n_extra} gambar kelebihan dari {len(names)}." if dup else f"Tidak ada gambar mirip dari {len(names)} gambar.",
                  assumptions=[f"Dua gambar dianggap mirip bila jarak Hamming ≤ {max_dist} dari 64 bit.", "Tahan terhadap perubahan ukuran/kompresi; tidak tahan terhadap rotasi/crop besar."],
                  tables=[table("Kelompok mirip", pd.DataFrame(rows))] if rows else [],
                  stats={"groups": len(dup), "redundant": n_extra, "group_files": dup},
                  actions=[{"type": "delete_rows_by_filename", "label": f"Hapus {n_extra} gambar kelebihan", "filenames": [n for g in dup for n in g[1:]]}] if n_extra else [])
