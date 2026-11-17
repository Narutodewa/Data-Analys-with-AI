"""Fitur ala spreadsheet: paging/sort/filter, kolom rumus, pivot, kueri terstruktur (aman, tanpa eksekusi kode bebas)."""
from __future__ import annotations

import re

import pandas as pd

from .store import Dataset
from .utils import df_to_records, is_numeric, jsonable

AGGS = {"sum", "mean", "median", "min", "max", "count", "nunique", "std"}
OPS = {"==", "!=", ">", ">=", "<", "<=", "contains", "in", "isnull", "notnull"}
_FORBIDDEN = re.compile(r"__|import|exec|eval|open\s*\(|lambda|@", re.I)


def page(df: pd.DataFrame, offset=0, limit=100, sort=None, desc=False, q=None) -> dict:
    d = df
    if q:
        mask = d.astype(str).apply(lambda c: c.str.contains(q, case=False, na=False, regex=False)).any(axis=1)
        d = d[mask]
    if sort:
        if sort not in d.columns:
            raise KeyError(f"Kolom '{sort}' tidak ada.")
        d = d.sort_values(sort, ascending=not desc, na_position="last", kind="stable")
    total = len(d)
    limit = max(1, min(int(limit), 500))
    chunk = d.iloc[max(0, offset): max(0, offset) + limit]
    return {"total": total, "offset": offset, "limit": limit, "columns": list(df.columns), "rows": df_to_records(chunk)}


def add_formula_column(ds: Dataset, name: str, expr: str) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Nama kolom baru wajib diisi.")
    if name in ds.df.columns:
        raise ValueError(f"Kolom '{name}' sudah ada.")
    if _FORBIDDEN.search(expr):
        raise ValueError("Ekspresi mengandung kata/simbol yang tidak diizinkan.")
    try:
        result = ds.df.eval(expr, engine="python")
    except Exception as e:
        raise ValueError(f"Ekspresi tidak valid: {e}. Contoh: harga * jumlah, atau `Total Harga` / 1000") from e
    if not isinstance(result, (pd.Series,)) :
        result = pd.Series([result] * len(ds.df), index=ds.df.index)
    ds.snapshot(f"Kolom baru '{name}' = {expr}")
    new = ds.df.copy()
    new[name] = result
    ds.df = new


def pivot(df: pd.DataFrame, index: list[str], columns: str | None, values: str | None, agg: str = "sum", limit: int = 200) -> dict:
    if agg not in AGGS:
        raise ValueError(f"Agregasi '{agg}' tidak dikenal.")
    for c in [*index, *(([columns]) if columns else []), *(([values]) if values else [])]:
        if c not in df.columns:
            raise KeyError(f"Kolom '{c}' tidak ada.")
    if not index:
        raise ValueError("Pilih minimal satu kolom baris (index).")
    if agg == "count" or not values:
        pt = df.pivot_table(index=index, columns=columns, values=values or index[0], aggfunc="count", fill_value=0) if columns \
            else df.groupby(index).size().rename("count").to_frame()
    else:
        if agg not in ("count", "nunique") and not is_numeric(df[values]):
            raise ValueError(f"Kolom '{values}' bukan numerik; gunakan agregasi 'count' atau 'nunique'.")
        pt = df.pivot_table(index=index, columns=columns, values=values, aggfunc=agg, fill_value=0 if agg in ("sum", "count", "nunique") else None)
    pt = pt.reset_index()
    pt.columns = [" / ".join(str(x) for x in c if str(x)) if isinstance(c, tuple) else str(c) for c in pt.columns]
    return {"columns": list(pt.columns), "rows": [dict(zip(pt.columns, [jsonable(v) for v in r])) for r in pt.head(limit).itertuples(index=False, name=None)], "total": len(pt)}


def run_query(df: pd.DataFrame, spec: dict, limit: int = 200) -> pd.DataFrame:
    """spec: {filters:[{col,op,value}], groupby:[..], agg:{col:fn}, sort:{by,desc}, limit}"""
    d = df
    for f in spec.get("filters") or []:
        col, op, val = f.get("col"), f.get("op"), f.get("value")
        if col not in d.columns or op not in OPS:
            raise ValueError(f"Filter tidak valid: {f}")
        s = d[col]
        if op == "isnull":
            m = s.isna()
        elif op == "notnull":
            m = s.notna()
        elif op == "contains":
            m = s.astype(str).str.contains(str(val), case=False, na=False, regex=False)
        elif op == "in":
            m = s.astype(str).isin([str(v) for v in (val if isinstance(val, list) else [val])])
        else:
            v = pd.to_numeric(val, errors="coerce") if is_numeric(s) else val
            m = {"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
        d = d[m.fillna(False)]
    gb, agg = spec.get("groupby") or [], spec.get("agg") or {}
    if gb:
        for c in gb:
            if c not in d.columns:
                raise ValueError(f"Kolom group '{c}' tidak ada.")
        if not agg:
            out = d.groupby(gb).size().rename("count").reset_index()
        else:
            for c, fn in agg.items():
                if c not in d.columns or fn not in AGGS:
                    raise ValueError(f"Agregasi tidak valid: {c}:{fn}")
            out = d.groupby(gb).agg(agg).reset_index()
            out.columns = [c if isinstance(c, str) else "_".join(map(str, c)) for c in out.columns]
        d = out
    srt = spec.get("sort") or {}
    if srt.get("by") in d.columns:
        d = d.sort_values(srt["by"], ascending=not srt.get("desc", True))
    return d.head(min(int(spec.get("limit") or limit), limit))
