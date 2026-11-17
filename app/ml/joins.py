"""Saran join antar dataset (tumpang tindih nilai) dan penggabungan."""
from __future__ import annotations

import pandas as pd

from ..utils import infer_kind, is_numeric
from .common import need, result, table


def _key(s: pd.Series) -> pd.Series:
    return s.dropna().astype(str).str.strip().str.lower()


def suggest(datasets: list, limit: int = 15) -> dict:
    cands = []
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            for ca in a.df.columns:
                sa = _key(a.df[ca])
                if len(sa) < 5 or infer_kind(a.df[ca]) in ("text",) or (is_numeric(a.df[ca]) and sa.nunique() < 5):
                    continue
                ua = set(sa.unique())
                for cb in b.df.columns:
                    if is_numeric(a.df[ca]) != is_numeric(b.df[cb]):
                        continue
                    sb = _key(b.df[cb])
                    if len(sb) < 5:
                        continue
                    ub = set(sb.unique())
                    inter = len(ua & ub)
                    if not inter:
                        continue
                    jac, cont = inter / len(ua | ub), inter / min(len(ua), len(ub))
                    name_bonus = 0.15 if str(ca).lower() == str(cb).lower() else 0.0
                    score = 0.6 * cont + 0.4 * jac + name_bonus
                    if cont >= 0.5 and score >= 0.45:
                        ka, kb = sa.is_unique, sb.is_unique
                        rel = "1:1" if ka and kb else "1:N" if ka else "N:1" if kb else "N:N"
                        cands.append({"kiri": a.name, "kolom_kiri": ca, "kanan": b.name, "kolom_kanan": cb, "relasi": rel, "kecocokan": round(cont, 3), "jaccard": round(jac, 3),
                                      "skor": round(score, 3), "id_kiri": a.id, "id_kanan": b.id})
    cands.sort(key=lambda r: -r["skor"])
    cands = cands[:limit]
    if not cands:
        return result("Saran join", "tumpang tindih nilai kolom", "Tidak ditemukan pasangan kolom yang cocok antar dataset.", assumptions=["Butuh ≥2 dataset dengan kolom kunci yang nilainya beririsan."])
    warn = [f"Relasi N:N pada {c['kolom_kiri']}↔{c['kolom_kanan']} akan melipatgandakan baris." for c in cands if c["relasi"] == "N:N"][:2]
    return result("Saran join", "tumpang tindih nilai (containment + Jaccard) + kemiripan nama", f"{len(cands)} kandidat kunci join ditemukan; teratas: {cands[0]['kiri']}.{cands[0]['kolom_kiri']} ↔ {cands[0]['kanan']}.{cands[0]['kolom_kanan']} ({cands[0]['relasi']}).",
                  assumptions=["Kecocokan = irisan nilai unik / nilai unik terkecil. Periksa makna kolom sebelum menggabungkan."], warnings=warn,
                  tables=[table("Kandidat join", pd.DataFrame(cands).drop(columns=["id_kiri", "id_kanan"]))], stats={"candidates": cands},
                  actions=[{"type": "merge", "label": f"Gabungkan {c['kiri']} + {c['kanan']} (left join pada {c['kolom_kiri']}={c['kolom_kanan']})", "left_id": c["id_kiri"], "right_id": c["id_kanan"], "left_on": c["kolom_kiri"], "right_on": c["kolom_kanan"], "how": "left"} for c in cands[:3]])


def merge(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str, how: str = "left") -> pd.DataFrame:
    need(how in ("left", "inner", "outer", "right"), "Jenis join tidak dikenal.")
    need(left_on in left.columns and right_on in right.columns, "Kolom kunci tidak ditemukan.")
    l, r = left.copy(), right.copy()
    l["__k"], r["__k"] = _key(l[left_on]).reindex(l.index), _key(r[right_on]).reindex(r.index)
    r = r.drop(columns=[right_on] if right_on != left_on else [], errors="ignore")
    out = l.merge(r, on="__k", how=how, suffixes=("", "_kanan")).drop(columns="__k")
    need(len(out) <= max(5 * len(left), 1_000_000), "Hasil join terlalu besar (relasi N:N?). Periksa kolom kunci.")
    return out.reset_index(drop=True)
