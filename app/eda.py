"""Exploratory Data Analysis: profil kolom, statistik, korelasi."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import as_datetime, infer_kind, is_numeric, jsonable


def column_profile(s: pd.Series, name: str) -> dict:
    n = len(s)
    kind = infer_kind(s)
    p = {
        "name": name, "dtype": str(s.dtype), "kind": kind,
        "missing": int(s.isna().sum()), "missing_pct": round(100 * s.isna().mean(), 1) if n else 0,
        "unique": int(s.nunique(dropna=True)),
    }
    c = s.dropna()
    if kind == "numeric" and len(c):
        c = c.astype("float64")
        q1, med, q3 = c.quantile([0.25, 0.5, 0.75])
        p.update({
            "mean": float(c.mean()), "std": float(c.std()) if len(c) > 1 else 0.0, "min": float(c.min()),
            "q1": float(q1), "median": float(med), "q3": float(q3), "max": float(c.max()),
            "skew": float(c.skew()) if len(c) > 2 else 0.0,
            "kurtosis": float(c.kurt()) if len(c) > 3 else 0.0,
            "zeros": int((c == 0).sum()),
        })
        counts, edges = np.histogram(c, bins=min(20, max(5, int(np.sqrt(len(c))))))
        p["hist"] = {"counts": counts.tolist(), "edges": edges.tolist()}
    elif kind in ("categorical", "boolean") and len(c):
        vc = c.astype(str).value_counts().head(10)
        p["top"] = [{"value": k, "count": int(v), "pct": round(100 * v / len(c), 1)} for k, v in vc.items()]
    elif kind == "datetime" and len(c):
        d = as_datetime(c).dropna()
        if len(d):
            p.update({"min": d.min().isoformat(), "max": d.max().isoformat(), "span_days": int((d.max() - d.min()).days)})
    elif kind == "text" and len(c):
        ln = c.astype(str).str.len()
        p.update({"avg_len": float(ln.mean()), "max_len": int(ln.max())})
    return p


def correlations(df: pd.DataFrame, max_cols: int = 30) -> dict:
    num = df.select_dtypes(include="number").loc[:, lambda d: d.nunique() > 1]
    if num.shape[1] < 2:
        return {"columns": [], "matrix": [], "top_pairs": []}
    num = num.iloc[:, :max_cols]
    corr = num.corr(numeric_only=True)
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append({"a": cols[i], "b": cols[j], "r": float(v)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {"columns": cols, "matrix": jsonable(corr.values), "top_pairs": pairs[:10]}


def run_eda(df: pd.DataFrame) -> dict:
    n = len(df)
    cols = [column_profile(df[c], c) for c in df.columns]
    kinds = pd.Series([c["kind"] for c in cols]).value_counts().to_dict()
    total_cells = max(n * df.shape[1], 1)
    return jsonable({
        "overview": {
            "rows": n, "cols": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": round(100 * df.isna().sum().sum() / total_cells, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
            "kinds": kinds,
        },
        "columns": cols,
        "correlations": correlations(df),
    })
