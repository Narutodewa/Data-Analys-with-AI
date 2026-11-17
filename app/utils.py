"""Helper: serialisasi JSON aman (NaN/numpy/datetime) dan inferensi tipe kolom."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd


def jsonable(o):
    if o is None or o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, str):
        return o
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, (pd.Timedelta, dt.timedelta)):
        return str(o)
    if isinstance(o, np.ndarray):
        return [jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(x) for x in o]
    if isinstance(o, pd.Series):
        return [jsonable(x) for x in o.tolist()]
    return str(o)


def df_to_records(df: pd.DataFrame) -> list[dict]:
    cols = list(df.columns)
    return [
        {"_row": jsonable(idx), **{c: jsonable(v) for c, v in zip(cols, row)}}
        for idx, row in zip(df.index, df.itertuples(index=False, name=None))
    ]


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def as_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", format="mixed")


def _looks_datetime(s: pd.Series) -> bool:
    if is_numeric(s) or pd.api.types.is_bool_dtype(s):
        return False
    sample = s.dropna().astype(str).head(200)
    if len(sample) < 3:
        return False
    if sample.str.contains(r"[-/:]", regex=True).mean() < 0.8:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def infer_kind(s: pd.Series) -> str:
    """numeric | boolean | datetime | categorical | text"""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if is_numeric(s):
        vals = set(s.dropna().unique().tolist())
        if vals and vals <= {0, 1} and len(s.dropna()) > 2:
            return "boolean"
        return "numeric"
    if _looks_datetime(s):
        return "datetime"
    n = s.notna().sum()
    nun = s.nunique(dropna=True)
    if n == 0:
        return "categorical"
    if nun <= 50 or nun / n <= 0.5:
        return "categorical"
    return "text"
