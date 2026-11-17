"""Anomali multivariat: Isolation Forest, Local Outlier Factor, dan MAD (robust z-score)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from .common import layout, need, numeric_cols, result, robust_z, table

METHODS = {"iforest": "Isolation Forest", "lof": "Local Outlier Factor", "mad": "MAD (robust z-score)"}


def score_rows(df: pd.DataFrame, columns=None, method="iforest", contamination=0.02):
    cols = numeric_cols(df, columns)
    need(len(cols) >= 1, "Butuh minimal satu kolom numerik.")
    need(len(df) >= 20, "Butuh minimal 20 baris untuk mendeteksi anomali.")
    need(method in METHODS, f"Metode harus salah satu dari {', '.join(METHODS)}.")
    contamination = float(min(max(contamination, 0.001), 0.3))
    X = df[cols].astype(float)
    Xf = X.fillna(X.median()).fillna(0.0)
    if method == "mad":
        z = Xf.apply(robust_z).abs()
        score = z.max(axis=1).to_numpy()
        flagged = score > 3.5
    else:
        need(len(cols) >= 2 or method == "iforest", "LOF paling bermakna dengan ≥2 kolom numerik.")
        Xs = RobustScaler().fit_transform(Xf)
        if method == "iforest":
            m = IsolationForest(n_estimators=200, contamination=contamination, random_state=0).fit(Xs)
            score = -m.score_samples(Xs)
            flagged = m.predict(Xs) == -1
        else:
            k = max(2, min(20, len(Xs) - 1))
            lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
            flagged = lof.fit_predict(Xs) == -1
            score = -lof.negative_outlier_factor_
    return cols, Xf, np.asarray(score, dtype=float), np.asarray(flagged, dtype=bool)


def detect(df, columns=None, method="iforest", contamination=0.02, limit=50) -> dict:
    cols, Xf, score, flagged = score_rows(df, columns, method, contamination)
    rz = Xf.apply(robust_z)
    rows = []
    for pos in np.argsort(-score)[: int(flagged.sum())][:limit]:
        idx = df.index[pos]
        dev = rz.iloc[pos].abs().sort_values(ascending=False).head(3)
        reasons = "; ".join(f"{c}={Xf.iloc[pos][c]:.4g} (z={rz.iloc[pos][c]:+.1f})" for c in dev.index if dev[c] >= 2) or "kombinasi nilai yang jarang"
        rows.append({"row": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "score": round(float(score[pos]), 4), "alasan": reasons})
    n_flag = int(flagged.sum())
    charts = []
    if len(cols) >= 2:
        p = PCA(n_components=2, random_state=0).fit(Xf)
        xy = p.transform(Xf)
        ev = p.explained_variance_ratio_
        traces = [
            {"type": "scattergl", "mode": "markers", "name": "normal", "x": xy[~flagged, 0], "y": xy[~flagged, 1], "marker": {"size": 5, "opacity": .55}},
            {"type": "scattergl", "mode": "markers", "name": "anomali", "x": xy[flagged, 0], "y": xy[flagged, 1], "marker": {"size": 9, "color": "#ef4444", "line": {"width": 1.5, "color": "#fff"}}},
        ]
        charts.append({"title": "Peta anomali (PCA 2D)", "figure": {"data": traces, "layout": layout(f"Anomali pada ruang PCA ({ev.sum():.0%} variasi)", f"PC1 ({ev[0]:.0%})", f"PC2 ({ev[1]:.0%})")}})
    ids = [r["row"] for r in rows]
    return result(
        f"Anomali ({METHODS[method]})", METHODS[method],
        f"{n_flag} dari {len(df)} baris ditandai anomali ({100 * n_flag / len(df):.1f}%) berdasarkan {len(cols)} kolom numerik.",
        assumptions=["Nilai kosong diisi median hanya untuk penilaian (data asli tidak diubah).", "Kolom diskalakan dengan RobustScaler.",
                     "Anomali ≠ salah: bisa kasus ekstrem yang valid. Tinjau sebelum menghapus."] + ([f"Proporsi anomali diperkirakan {contamination:.1%} (parameter)."] if method != "mad" else ["Ambang MAD: |z| > 3,5."]),
        tables=[table("Baris teranomali", pd.DataFrame(rows))] if rows else [], charts=charts,
        stats={"flagged": n_flag, "rows": len(df), "columns": cols, "row_ids": ids},
        actions=[{"type": "delete_rows", "label": f"Hapus {len(ids)} baris teratas", "rows": ids}] if ids else [])
