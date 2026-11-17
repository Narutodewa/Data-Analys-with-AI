"""Kontrak bersama semua mesin analitik.

Setiap mesin mengembalikan dict dengan bentuk sama (lihat `result`). Aturan emas:
angka dihitung di sini (statistik/ML), LLM hanya menarasikan hasilnya.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..charts import PALETTE
from ..utils import infer_kind, is_numeric, jsonable


class MissingDependency(ValueError):
    pass


def result(title, method, summary, *, assumptions=(), tables=(), charts=(), warnings=(), stats=None, actions=()):
    return jsonable({
        "title": title, "method": method, "summary": summary,
        "assumptions": list(assumptions), "warnings": list(warnings),
        "stats": stats or {}, "tables": list(tables), "charts": list(charts), "actions": list(actions),
    })


def table(title, df: pd.DataFrame, limit=50):
    d = df.head(limit)
    return {"title": title, "columns": [str(c) for c in d.columns],
            "rows": jsonable(d.astype(object).where(d.notna(), None).values.tolist()), "total": int(len(df))}


def layout(title, xt=None, yt=None, **kw):
    lay = {"title": {"text": title, "font": {"size": 14}}, "xaxis": {"title": xt}, "yaxis": {"title": yt},
           "margin": {"l": 55, "r": 20, "t": 44, "b": 50}, "colorway": PALETTE}
    lay.update(kw)
    return lay


def numeric_cols(df: pd.DataFrame, cols=None, max_cols=50, min_unique=2) -> list[str]:
    cand = [c for c in (cols or df.columns) if c in df.columns]
    out = [c for c in cand if is_numeric(df[c]) and infer_kind(df[c]) == "numeric" and df[c].nunique(dropna=True) >= min_unique]
    return out[:max_cols]


def need(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def import_or_fail(name: str, pip: str | None = None):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise MissingDependency(f"Fitur ini membutuhkan paket '{pip or name}'. Pasang dengan: pip install {pip or name}") from e


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        sd = s.std()
        return (s - med) / sd if sd else s * 0.0
    return 0.6745 * (s - med) / mad
