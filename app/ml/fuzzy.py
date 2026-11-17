"""Pengelompokan variasi penulisan kategori yang mirip (typo, singkatan) dengan fuzzy matching."""
from __future__ import annotations

import difflib
import re

import pandas as pd

try:
    from rapidfuzz import fuzz

    def _ratio(a, b):
        return fuzz.ratio(a, b)
except ImportError:  # pragma: no cover
    def _ratio(a, b):
        return 100 * difflib.SequenceMatcher(None, a, b).ratio()

MAX_UNIQUE = 300


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def variant_mapping(values: pd.Series, threshold: int = 88) -> dict[str, str]:
    """variasi -> bentuk kanonik (yang paling sering). Hanya menggabungkan nilai yang BEDA setelah normalisasi huruf/spasi."""
    vc = values.dropna().astype(str).value_counts()
    if not 1 < len(vc) <= MAX_UNIQUE:
        return {}
    items = list(vc.index)
    parent = {v: v for v in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(items):
        na = _norm(a)
        if len(na) < 4 or re.fullmatch(r"[\d\W_]+", na):
            continue
        for b in items[i + 1:]:
            nb = _norm(b)
            if na == nb or len(nb) < 4 or abs(len(na) - len(nb)) > 3:
                continue
            if re.search(r"\d", na) != re.search(r"\d", nb) or re.sub(r"\D", "", na) != re.sub(r"\D", "", nb):
                continue  # jangan gabungkan "Toko 1" dengan "Toko 2"
            if _ratio(na, nb) >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    groups: dict[str, list[str]] = {}
    for v in items:
        groups.setdefault(find(v), []).append(v)
    mapping = {}
    for members in groups.values():
        if len(members) > 1:
            top = max(members, key=lambda m: vc[m])
            for m in members:
                if m != top:
                    mapping[m] = top
    return mapping
