"""Imputasi berbasis model (KNN, iteratif/MICE) dan pemeriksaan pola data kosong."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

from .common import need, numeric_cols


def impute_column(df: pd.DataFrame, col: str, method: str = "knn") -> pd.Series:
    """Nilai terisi untuk SELURUH kolom `col` (nilai asli dipertahankan). Memakai kolom numerik lain sebagai prediktor."""
    need(col in numeric_cols(df, [col], min_unique=1) or pd.api.types.is_numeric_dtype(df[col]), f"Kolom '{col}' bukan numerik.")
    others = [c for c in numeric_cols(df, None) if c != col]
    need(others, "Imputasi model butuh minimal satu kolom numerik lain sebagai prediktor.")
    cols = [col] + others[:30]
    X = df[cols].astype(float)
    need(X[col].notna().sum() >= 10, "Terlalu sedikit nilai terisi (<10) untuk melatih imputer.")
    sc = StandardScaler()
    Z = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=cols)  # NaN dipertahankan
    if method == "knn":
        imp = KNNImputer(n_neighbors=min(5, max(1, int(X[col].notna().sum()) - 1)), weights="distance")
    elif method == "iterative":
        imp = IterativeImputer(random_state=0, max_iter=15, sample_posterior=False, skip_complete=True)
    else:
        raise ValueError("Metode imputasi harus 'knn' atau 'iterative'.")
    out = imp.fit_transform(Z)
    j = cols.index(col)
    filled = out[:, j] * sc.scale_[j] + sc.mean_[j]
    res = pd.Series(filled, index=df.index)
    res[df[col].notna()] = df.loc[df[col].notna(), col].astype(float)
    return res


def missingness_note(df: pd.DataFrame, col: str, alpha: float = 0.01) -> str | None:
    """Heuristik: apakah data kosong tampak acak (MCAR)? Uji Welch pada kolom numerik lain antara baris kosong vs terisi."""
    miss = df[col].isna()
    if miss.sum() < 10 or (~miss).sum() < 10:
        return None
    others = [c for c in numeric_cols(df) if c != col][:20]
    ps = []
    for c in others:
        a, b = df.loc[miss, c].dropna(), df.loc[~miss, c].dropna()
        if len(a) >= 5 and len(b) >= 5 and (a.std() > 0 or b.std() > 0):
            ps.append((stats.ttest_ind(a, b, equal_var=False).pvalue, c))
    if not ps:
        return None
    thr = alpha / len(ps)  # Bonferroni
    hit = [c for p, c in ps if p < thr]
    if hit:
        return f"Pola kosong tampak TIDAK acak: berkaitan dengan {', '.join(hit[:3])}. Isi dengan model (KNN/iteratif) lebih aman daripada rata-rata."
    return "Pola kosong tampak acak terhadap kolom numerik lain (uji Welch + Bonferroni); isi statistik sederhana relatif aman."
