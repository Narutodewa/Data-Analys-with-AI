"""Embedding semantik OPSIONAL (teks & gambar). Aktif bila paket tambahan terpasang: pip install -r requirements-extra.txt

Tidak diuji otomatis di CI karena model besar perlu diunduh; jalur 'paket tidak ada' diuji.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .common import import_or_fail, need, result, table

TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
IMAGE_MODEL = ("ViT-B-32", "laion2b_s34b_b79k")


def available() -> dict:
    import importlib.util as u
    return {"teks": u.find_spec("sentence_transformers") is not None, "gambar": u.find_spec("open_clip") is not None and u.find_spec("torch") is not None}


def _cluster_report(title, method, labels, names, extra_note, vecs):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    k = int(max(2, min(8, len(names) // 5)))
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(vecs)
    lab = km.labels_
    sil = float(silhouette_score(vecs, lab)) if len(set(lab)) > 1 else None
    rows = []
    for c in range(k):
        idx = np.where(lab == c)[0]
        d = np.linalg.norm(vecs[idx] - km.cluster_centers_[c], axis=1)
        rep = [names[i] for i in idx[np.argsort(d)[:3]]]
        rows.append({"kelompok": f"K{c + 1}", "ukuran": int(len(idx)), "contoh_representatif": " | ".join(str(r)[:60] for r in rep)})
    return result(title, method, f"{k} kelompok semantik dari {len(names)} item (silhouette {sil:.2f})." if sil is not None else f"{k} kelompok.", assumptions=[extra_note],
                  tables=[table("Kelompok", pd.DataFrame(rows))], stats={"k": k, "silhouette": sil, "labels": {str(n): int(l) for n, l in zip(names, lab)}})


def cluster_text(df: pd.DataFrame, column: str) -> dict:
    st = import_or_fail("sentence_transformers", "sentence-transformers")
    need(column in df.columns, f"Kolom '{column}' tidak ditemukan.")
    docs = df[column].dropna().astype(str)
    docs = docs[docs.str.len() >= 3].head(5000)
    need(len(docs) >= 10, "Butuh minimal 10 teks.")
    vecs = st.SentenceTransformer(TEXT_MODEL).encode(docs.tolist(), normalize_embeddings=True, show_progress_bar=False)
    return _cluster_report(f"Kelompok semantik '{column}'", f"embedding {TEXT_MODEL} + k-means", None, docs.tolist(), "Embedding multibahasa (termasuk Indonesia); model diunduh sekali dari Hugging Face.", np.asarray(vecs))


def cluster_images(ds) -> dict:
    torch = import_or_fail("torch")
    oc = import_or_fail("open_clip", "open_clip_torch")
    from PIL import Image
    imgs = ds.meta.get("images") or {}
    need(ds.kind == "image" and imgs, "Dataset ini bukan kumpulan gambar.")
    names = list(imgs)[:2000]
    model, _, pre = oc.create_model_and_transforms(IMAGE_MODEL[0], pretrained=IMAGE_MODEL[1])
    model.eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(names), 32):
            batch = torch.stack([pre(Image.open(io.BytesIO(imgs[n])).convert("RGB")) for n in names[i:i + 32]])
            v = model.encode_image(batch)
            vecs.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy())
    return _cluster_report("Kelompok visual gambar", f"CLIP {IMAGE_MODEL[0]} + k-means", None, names, "Embedding CLIP; model diunduh sekali.", np.vstack(vecs))
