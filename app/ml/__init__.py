"""Lapisan analitik: statistik klasik + ML. Satu pintu masuk `run()` dipakai oleh API dan oleh AI (LLM)."""
from __future__ import annotations

import pandas as pd

from . import cluster, embed, imgml, joins, outliers, pii, predict, sqlq, stats_tests, textmining, timeseries
from .common import MissingDependency, need, result, table

# type parameter: column:<jenis>|columns:<jenis>|select|number|text|bool ; jenis = numeric|datetime|text|any
CATALOG = {
    "insights": {"title": "Temuan terverifikasi statistik", "desc": "Uji hipotesis + koreksi FDR + effect size + mutual information. Hanya menampilkan temuan yang lolos.", "needs": "tabel", "params": []},
    "anomalies": {"title": "Deteksi anomali", "desc": "Baris yang janggal bila dilihat dari gabungan kolom (Isolation Forest, LOF, MAD).", "needs": "tabel",
                  "params": [{"name": "method", "type": "select", "options": ["iforest", "lof", "mad"], "default": "iforest", "label": "Metode"},
                             {"name": "contamination", "type": "number", "default": 0.02, "label": "Perkiraan proporsi anomali"},
                             {"name": "columns", "type": "columns:numeric", "label": "Kolom (kosong = semua numerik)"}]},
    "segment": {"title": "Segmentasi / klaster", "desc": "Kelompokkan baris yang mirip; k dipilih otomatis (silhouette) atau HDBSCAN.", "needs": "tabel",
                "params": [{"name": "method", "type": "select", "options": ["kmeans", "hdbscan"], "default": "kmeans", "label": "Metode"},
                           {"name": "k", "type": "number", "label": "Jumlah segmen (kosong = otomatis)"},
                           {"name": "columns", "type": "columns:numeric", "label": "Kolom (kosong = semua numerik)"}]},
    "timeseries": {"title": "Deret waktu", "desc": "Tren, musiman, titik perubahan, dan ramalan dengan backtest vs baseline naif.", "needs": "tabel",
                   "params": [{"name": "date", "type": "column:datetime", "required": True, "label": "Kolom tanggal"}, {"name": "value", "type": "column:numeric", "required": True, "label": "Kolom nilai"},
                              {"name": "agg", "type": "select", "options": ["sum", "mean", "median", "min", "max", "count"], "default": "sum", "label": "Agregasi"}, {"name": "horizon", "type": "number", "label": "Periode ramalan (kosong = otomatis)"}]},
    "predict": {"title": "Prediksi & faktor berpengaruh", "desc": "AutoML ringan: baseline vs model, validasi silang, permutation importance, deteksi kebocoran data.", "needs": "tabel",
                "params": [{"name": "target", "type": "column:any", "required": True, "label": "Kolom target"}, {"name": "features", "type": "columns:any", "label": "Fitur (kosong = semua)"}]},
    "pii": {"title": "Pemindai data pribadi (PII)", "desc": "Email, telepon, NIK, NPWP, kartu kredit, IP, dan nama kolom sensitif.", "needs": "tabel", "params": []},
    "text": {"title": "Analisis teks", "desc": "Topik (TF-IDF+NMF), kata kunci, sentimen leksikon.", "needs": "tabel", "params": [{"name": "column", "type": "column:text", "required": True, "label": "Kolom teks"}]},
    "images_dup": {"title": "Gambar duplikat / mirip", "desc": "Perceptual hash (dHash) + jarak Hamming.", "needs": "citra", "params": [{"name": "max_dist", "type": "number", "default": 6, "label": "Jarak Hamming maks (0–64)"}]},
    "embed_text": {"title": "Kelompok semantik teks (embedding)", "desc": "OPSIONAL: butuh sentence-transformers.", "needs": "tabel", "optional": "teks", "params": [{"name": "column", "type": "column:text", "required": True, "label": "Kolom teks"}]},
    "embed_images": {"title": "Kelompok visual gambar (CLIP)", "desc": "OPSIONAL: butuh open_clip_torch + torch.", "needs": "citra", "optional": "gambar", "params": []},
    "joins": {"title": "Saran join antar dataset", "desc": "Kolom kunci yang cocok di dataset lain.", "needs": "semua", "params": []},
    "sql": {"title": "SQL (baca-saja)", "desc": "Kueri SELECT lintas semua dataset.", "needs": "semua", "params": [{"name": "sql", "type": "text", "required": True, "label": "SELECT …"}]},
}


def catalog() -> list[dict]:
    av = embed.available()
    return [{"id": k, **v, "available": (av.get(v["optional"], False) if v.get("optional") else True)} for k, v in CATALOG.items()]


def _insights(df) -> dict:
    ins = stats_tests.significant_insights(df)
    rows = pd.DataFrame([{"temuan": i["title"], "metode": i["method"], "effect": round(i["effect"], 3) if i.get("effect") is not None else None} for i in ins])
    return result("Temuan terverifikasi statistik", "uji hipotesis + BH-FDR + effect size + MI",
                  f"{len(ins)} temuan lolos uji (q<0,05 dan effect size bermakna)." if ins else "Tidak ada hubungan yang lolos uji statistik dengan effect size bermakna.",
                  assumptions=["Semua uji dijalankan, lalu p-value dikoreksi serentak (Benjamini-Hochberg) agar temuan kebetulan tidak lolos.", "Uji non-parametrik (Spearman, Mann-Whitney/Kruskal) dipilih agar tahan terhadap data miring/outlier.", "Signifikan ≠ penting dan ≠ sebab-akibat; lihat effect size."],
                  tables=[table("Temuan", rows)] if len(rows) else [], stats={"count": len(ins)})


def run(name: str, ds, params: dict | None, all_datasets: list) -> dict:
    p = dict(params or {})
    need(name in CATALOG, f"Mesin '{name}' tidak dikenal. Pilihan: {', '.join(CATALOG)}")
    if name == "joins":
        return joins.suggest(all_datasets)
    if name == "sql":
        return sqlq.query(all_datasets, str(p.get("sql", "")))
    need(ds is not None, "Pilih dataset terlebih dahulu.")
    df = ds.df
    cols = p.get("columns") or None
    if name == "insights":
        return _insights(df)
    if name == "anomalies":
        return outliers.detect(df, cols, p.get("method", "iforest"), float(p.get("contamination") or 0.02))
    if name == "segment":
        k = p.get("k")
        out = cluster.segment(df, cols, p.get("method", "kmeans"), int(k) if k else None, bool(p.get("apply")))
        return out
    if name == "timeseries":
        h = p.get("horizon")
        return timeseries.analyze(df, p.get("date"), p.get("value"), p.get("agg", "sum"), int(h) if h else None)
    if name == "predict":
        return predict.automl(df, p.get("target"), p.get("features") or None)
    if name == "pii":
        f = pii.scan(df)
        return result("Pemindai data pribadi (PII)", "pola nilai (regex + Luhn) + petunjuk nama kolom",
                      f"{len(f)} kolom diduga berisi data pribadi." if f else "Tidak ada kolom yang tampak berisi data pribadi.",
                      assumptions=["Deteksi berbasis pola; bisa meleset (false positive/negative). Tinjau manual sebelum membagikan data."],
                      tables=[table("Kolom sensitif", pd.DataFrame(f))] if f else [], stats={"findings": f})
    if name == "text":
        return textmining.analyze(df, p.get("column"))
    if name == "images_dup":
        return imgml.near_duplicates(ds, int(p.get("max_dist") if p.get("max_dist") is not None else 6))
    if name == "embed_text":
        return embed.cluster_text(df, p.get("column"))
    if name == "embed_images":
        return embed.cluster_images(ds)
    raise ValueError("Mesin belum diimplementasikan.")


__all__ = ["run", "catalog", "CATALOG", "MissingDependency"]
