"""Deteksi data pribadi (PII) berbasis pola + petunjuk nama kolom, serta penyamaran/pseudonim."""
from __future__ import annotations

import hashlib
import re

import pandas as pd

PATTERNS = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$"),
    "telepon": re.compile(r"^(\+?62|0)8\d{8,11}$"),
    "NIK": re.compile(r"^(1[1-9]|[3-9]\d)\d{14}$"),
    "NPWP": re.compile(r"^\d{2}\.\d{3}\.\d{3}\.\d-\d{3}\.\d{3}$|^\d{15,16}$"),
    "alamat IP": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
}
NAME_HINTS = {
    "nama": "nama orang", "name": "nama orang", "alamat": "alamat", "address": "alamat", "email": "email", "telepon": "telepon",
    "phone": "telepon", "hp": "telepon", "nik": "NIK", "ktp": "NIK", "npwp": "NPWP", "password": "kata sandi", "kata_sandi": "kata sandi",
}


def _luhn(num: str) -> bool:
    d = [int(c) for c in num][::-1]
    return (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0 if len(d) >= 13 else False


def scan_column(s: pd.Series, name: str) -> dict | None:
    v = s.dropna().astype(str).str.strip()
    if len(v) < 3:
        return None
    sample = v.sample(min(len(v), 2000), random_state=0)
    best = None
    for kind, pat in PATTERNS.items():
        share = float(sample.map(lambda x: bool(pat.match(x))).mean())
        if share >= 0.3 and (best is None or share > best[1]):
            best = (kind, share)
    digits = sample.str.replace(r"[\s-]", "", regex=True)
    cc = float(digits.map(lambda x: x.isdigit() and 13 <= len(x) <= 19 and _luhn(x)).mean())
    if cc >= 0.8 and (best is None or cc >= best[1]):
        best = ("kartu kredit", cc)
    hint = next((h for k, h in NAME_HINTS.items() if re.search(rf"(^|[_\W]){k}($|[_\W])", name.lower())), None)
    if best:
        return {"column": name, "type": best[0], "share": round(best[1], 3), "via": "pola nilai", "example": mask_value(sample.iloc[0], best[0])}
    if hint and v.nunique() / len(v) > 0.3 and hint != "kata sandi":
        return {"column": name, "type": hint, "share": None, "via": "nama kolom", "example": mask_value(sample.iloc[0], hint)}
    if hint == "kata sandi":
        return {"column": name, "type": hint, "share": None, "via": "nama kolom", "example": "***"}
    return None


def mask_value(x, kind="") -> str:
    s = str(x)
    if kind == "email" and "@" in s:
        u, dom = s.split("@", 1)
        return (u[:1] + "***@" + dom)
    if len(s) <= 3:
        return "*" * len(s)
    if kind in ("telepon", "NIK", "NPWP", "kartu kredit"):
        return "*" * (len(s) - 3) + s[-3:]
    return s[0] + "*" * min(len(s) - 1, 6)


def pseudonym(x, salt="data-analyst") -> str:
    return "anon_" + hashlib.sha256((salt + str(x)).encode()).hexdigest()[:10]


def scan(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in df.columns:
        if df[c].dtype.kind in "fb":  # float/bool bukan PII
            continue
        f = scan_column(df[c], str(c))
        if f:
            out.append(f)
    return out
