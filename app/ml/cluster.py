"""Segmentasi: k-means (k dipilih lewat silhouette) atau HDBSCAN, dengan profil segmen dan peta PCA."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import layout, need, numeric_cols, result, table


def _prep(df, columns):
    cols = numeric_cols(df, columns)
    need(len(cols) >= 2, "Segmentasi butuh minimal 2 kolom numerik.")
    need(len(df) >= 30, "Butuh minimal 30 baris untuk segmentasi.")
    X = df[cols].astype(float)
    Xf = X.fillna(X.median()).fillna(0.0)
    return cols, Xf, StandardScaler().fit_transform(Xf)


def labels_for(df, columns=None, method="kmeans", k=None):
    cols, Xf, Z = _prep(df, columns)
    rng = np.random.default_rng(0)
    sil_idx = rng.choice(len(Z), min(len(Z), 4000), replace=False)
    info = {}
    if method == "kmeans":
        if k:
            best_k = int(max(2, min(k, len(Z) - 1)))
        else:
            scores = {}
            for kk in range(2, min(9, len(Z) // 10 + 2)):
                lab = KMeans(n_clusters=kk, n_init=5, random_state=0).fit_predict(Z)
                scores[kk] = silhouette_score(Z[sil_idx], lab[sil_idx])
            best_k = max(scores, key=scores.get)
            info["silhouette_by_k"] = {int(a): round(float(b), 3) for a, b in scores.items()}
        lab = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(Z)
    elif method == "hdbscan":
        lab = HDBSCAN(min_cluster_size=max(5, len(Z) // 50), copy=True).fit_predict(Z)
    else:
        raise ValueError("Metode harus 'kmeans' atau 'hdbscan'.")
    valid = lab >= 0
    sil = None
    if len(set(lab[valid])) >= 2 and valid.sum() > 5:
        idx = sil_idx[valid[sil_idx]]
        if len(set(lab[idx])) >= 2:
            sil = float(silhouette_score(Z[idx], lab[idx]))
    return cols, Xf, Z, lab, sil, info


def segment(df, columns=None, method="kmeans", k=None, apply=False, name="segmen") -> dict:
    cols, Xf, Z, lab, sil, info = labels_for(df, columns, method, k)
    uniq = sorted(set(lab))
    n_cl = len([u for u in uniq if u >= 0])
    noise = int((lab < 0).sum())
    overall_mean, overall_sd = Xf.mean(), Xf.std().replace(0, 1)
    rows = []
    for u in uniq:
        m = lab == u
        z = ((Xf[m].mean() - overall_mean) / overall_sd)
        top = z.abs().sort_values(ascending=False).head(3).index
        desc = "; ".join(f"{c} {'tinggi' if z[c] > 0 else 'rendah'} ({Xf[m][c].mean():.4g} vs {overall_mean[c]:.4g})" for c in top if abs(z[c]) >= 0.3) or "mendekati rata-rata"
        rows.append({"segmen": ("noise" if u < 0 else f"S{u + 1}"), "ukuran": int(m.sum()), "persen": round(100 * m.mean(), 1), "ciri_khas": desc})
    pca = PCA(n_components=2, random_state=0).fit(Z)
    xy = pca.transform(Z)
    traces = [{"type": "scattergl", "mode": "markers", "name": ("noise" if u < 0 else f"S{u + 1}"), "x": xy[lab == u, 0], "y": xy[lab == u, 1], "marker": {"size": 6, "opacity": .7}} for u in uniq]
    ev = pca.explained_variance_ratio_
    warnings = []
    if sil is None:
        interp = "Silhouette tidak dapat dihitung."
    elif sil >= 0.5:
        interp = f"silhouette {sil:.2f}: segmen terpisah jelas."
    elif sil >= 0.25:
        interp = f"silhouette {sil:.2f}: segmen cukup terpisah."
    else:
        interp = f"silhouette {sil:.2f}: segmen lemah/tumpang tindih."
        warnings.append("Segmen tidak terpisah kuat; hati-hati menafsirkannya sebagai kelompok yang nyata.")
    if noise:
        warnings.append(f"{noise} baris dianggap noise (tidak masuk segmen mana pun).")
    summary = f"{n_cl} segmen ditemukan dari {len(Xf)} baris memakai {len(cols)} kolom ({interp})"
    actions = [{"type": "add_column", "label": f"Tambahkan kolom '{name}' ke data", "params": {"apply": True}}]
    out = result("Segmentasi", "k-means + silhouette" if method == "kmeans" else "HDBSCAN", summary,
                 assumptions=["Kolom numerik distandarkan (z-score); nilai kosong diisi median untuk segmentasi.", "Segmen adalah pola statistik, bukan kebenaran mutlak."],
                 warnings=warnings, tables=[table("Profil segmen", pd.DataFrame(rows))],
                 charts=[{"title": "Peta segmen (PCA 2D)", "figure": {"data": traces, "layout": layout(f"Segmen pada ruang PCA ({ev.sum():.0%} variasi)", f"PC1 ({ev[0]:.0%})", f"PC2 ({ev[1]:.0%})")}}],
                 stats={"n_clusters": n_cl, "silhouette": sil, "noise": noise, "columns": cols, **info}, actions=actions)
    if apply:
        out["labels"] = {str(i): ("noise" if v < 0 else f"S{v + 1}") for i, v in zip(df.index, lab)}
    return out
