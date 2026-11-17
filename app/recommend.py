"""Rekomendasi analisis (heuristik, tanpa AI): temukan pola/masalah dan usulkan diagram.

- Mode "cari masalah": goal kosong -> pindai data untuk temuan menarik.
- Mode "ada masalah": goal berisi kalimat bebas -> prioritaskan kolom/jenis analisis yang relevan.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .charts import build_chart
from .utils import as_datetime, infer_kind, is_numeric

KEYWORDS = {
    "trend": ["tren", "trend", "waktu", "bulan", "tahun", "harian", "naik", "turun", "over time", "musim", "growth", "pertumbuhan"],
    "relation": ["hubungan", "korelasi", "pengaruh", "berkaitan", "relationship", "correlat", "faktor", "berdampak"],
    "compare": ["bandingkan", "perbandingan", "per ", "antar", "tertinggi", "terendah", "terbaik", "terburuk", "compare", "top", "kategori", "segmen"],
    "distribution": ["distribusi", "sebaran", "persebaran", "range", "rentang", "distribution", "outlier", "ekstrem"],
    "composition": ["komposisi", "proporsi", "persentase", "porsi", "share", "pangsa"],
}


def _mentioned_columns(goal: str, cols) -> list[str]:
    g = goal.lower()
    return [c for c in cols if re.search(rf"(?<![\w]){re.escape(str(c).lower().replace('_', ' '))}(?![\w])", g.replace("_", " "))]


def _intents(goal: str) -> set[str]:
    g = goal.lower()
    return {k for k, words in KEYWORDS.items() if any(w in g for w in words)}


def find_insights(df: pd.DataFrame) -> list[dict]:
    """Temuan otomatis, diurutkan dari yang paling penting."""
    out: list[dict] = []
    n = len(df)
    kinds = {c: infer_kind(df[c]) for c in df.columns}
    nums = [c for c, k in kinds.items() if k == "numeric"]
    cats = [c for c, k in kinds.items() if k == "categorical" and 1 < df[c].nunique() <= 30]
    dts = [c for c, k in kinds.items() if k == "datetime"]

    def add(score, title, detail, spec=None, kind="insight"):
        out.append({"score": float(score), "kind": kind, "title": title, "detail": detail, "spec": spec})

    # kualitas data
    miss = df.isna().mean()
    for c in miss[miss >= 0.2].index:
        add(0.5 + miss[c] / 2, f"'{c}' banyak yang kosong ({miss[c]:.0%})", "Hasil analisis yang memakai kolom ini bisa bias. Tinjau di tab Cleaning.", kind="quality")

    # korelasi kuat
    if len(nums) >= 2:
        corr = df[nums].corr()
        seen = set()
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) >= 0.6 and (a, b) not in seen:
                    seen.add((a, b))
                    arah = "searah" if r > 0 else "berlawanan arah"
                    add(abs(r), f"'{a}' dan '{b}' berkorelasi kuat (r = {r:.2f})",
                        f"Keduanya bergerak {arah}. Korelasi bukan bukti sebab-akibat; periksa variabel lain.",
                        {"type": "scatter", "x": a, "y": b})

    # tren waktu
    for dc in dts[:1]:
        d = as_datetime(df[dc])
        for nc in nums[:6]:
            tmp = pd.DataFrame({"t": d, "v": df[nc]}).dropna().sort_values("t")
            if len(tmp) >= 8 and tmp["v"].std() > 0:
                x = (tmp["t"] - tmp["t"].min()).dt.total_seconds().to_numpy()
                if x.std() == 0:
                    continue
                r = np.corrcoef(x, tmp["v"].to_numpy(dtype=float))[0, 1]
                if abs(r) >= 0.3:
                    add(0.4 + abs(r) / 2, f"'{nc}' cenderung {'naik' if r > 0 else 'turun'} seiring waktu ({dc})",
                        f"Korelasi linear terhadap waktu r = {r:.2f}.", {"type": "line", "x": dc, "y": nc, "agg": "mean"})

    # perbedaan antar segmen (eta^2)
    for cc in cats[:6]:
        for nc in nums[:6]:
            tmp = df[[cc, nc]].dropna()
            if len(tmp) < 20 or tmp[nc].var() == 0:
                continue
            gm = tmp.groupby(cc)[nc].agg(["mean", "count"])
            gm = gm[gm["count"] >= 3]
            if len(gm) < 2:
                continue
            ssb = (gm["count"] * (gm["mean"] - tmp[nc].mean()) ** 2).sum()
            eta = ssb / ((tmp[nc] - tmp[nc].mean()) ** 2).sum()
            if eta >= 0.14:
                hi, lo = gm["mean"].idxmax(), gm["mean"].idxmin()
                add(0.3 + eta, f"'{nc}' berbeda jauh antar '{cc}'",
                    f"Rata-rata tertinggi di '{hi}' ({gm['mean'].max():.4g}), terendah di '{lo}' ({gm['mean'].min():.4g}). Kategori menjelaskan ±{eta:.0%} variasi.",
                    {"type": "bar", "x": cc, "y": nc, "agg": "mean"})

    # distribusi miring / outlier
    for c in nums:
        s = df[c].dropna()
        if len(s) >= 8:
            sk = s.skew()
            if abs(sk) >= 1:
                add(0.3 + min(abs(sk), 4) / 10, f"Distribusi '{c}' {'miring ke kanan' if sk > 0 else 'miring ke kiri'} (skew {sk:.1f})",
                    "Rata-rata bisa menyesatkan; pertimbangkan median atau transformasi log.", {"type": "histogram", "x": c})
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if iqr > 0 and n_out / len(s) >= 0.02:
                add(0.35 + min(n_out / len(s), 0.2), f"'{c}' punya {n_out} outlier ({n_out / len(s):.0%})", "Nilai ekstrem dapat mendominasi rata-rata.", {"type": "box", "y": c})

    # kategori dominan
    for c in cats:
        vc = df[c].value_counts(normalize=True)
        if vc.iloc[0] >= 0.7:
            add(0.3 + vc.iloc[0] / 4, f"'{c}' didominasi '{vc.index[0]}' ({vc.iloc[0]:.0%})", "Data tidak seimbang; perbandingan antar kelompok kecil kurang andal.", {"type": "pie", "x": c, "agg": "count"})

    out.sort(key=lambda r: -r["score"])
    return out


def suggest(df: pd.DataFrame, goal: str | None = None, max_charts: int = 6) -> dict:
    goal = (goal or "").strip()
    insights = find_insights(df)
    kinds = {c: infer_kind(df[c]) for c in df.columns}
    nums = [c for c, k in kinds.items() if k == "numeric"]
    cats = [c for c, k in kinds.items() if k == "categorical" and 1 < df[c].nunique() <= 30]
    dts = [c for c, k in kinds.items() if k == "datetime"]
    specs: list[tuple[str, str, dict]] = []  # (judul, alasan, spec)

    if goal:
        mentioned = _mentioned_columns(goal, df.columns)
        intents = _intents(goal)
        mn = [c for c in mentioned if c in nums] or nums[:1]
        mc = [c for c in mentioned if c in cats] or cats[:1]
        md = [c for c in mentioned if c in dts] or dts[:1]
        if "trend" in intents and md and mn:
            specs.append((f"Tren {mn[0]} menurut {md[0]}", "Anda menanyakan perubahan dari waktu ke waktu.", {"type": "line", "x": md[0], "y": mn[0], "agg": "sum"}))
        if "relation" in intents and len(mentioned) >= 2 and all(c in nums for c in mentioned[:2]):
            specs.append((f"Hubungan {mentioned[0]} vs {mentioned[1]}", "Anda menanyakan hubungan antar variabel.", {"type": "scatter", "x": mentioned[0], "y": mentioned[1]}))
        elif "relation" in intents and len(nums) >= 2:
            specs.append(("Peta korelasi", "Anda menanyakan hubungan antar variabel.", {"type": "heatmap"}))
        if ("compare" in intents or not intents) and mc and mn:
            specs.append((f"{mn[0]} per {mc[0]}", "Perbandingan antar kelompok.", {"type": "bar", "x": mc[0], "y": mn[0], "agg": "sum"}))
        if "distribution" in intents and mn:
            specs.append((f"Distribusi {mn[0]}", "Anda menanyakan sebaran nilai.", {"type": "histogram", "x": mn[0]}))
            specs.append((f"Sebaran {mn[0]} (box plot)", "Memperlihatkan outlier.", {"type": "box", "y": mn[0], "x": mc[0] if mc else None}))
        if "composition" in intents and mc:
            specs.append((f"Komposisi {mc[0]}", "Anda menanyakan proporsi.", {"type": "pie", "x": mc[0], "y": mn[0] if mn else None, "agg": "sum" if mn else "count"}))
        # temuan terkait kolom yang disebut
        if mentioned:
            related = [i for i in insights if any(f"'{c}'" in i["title"] for c in mentioned)]
            insights = related + [i for i in insights if i not in related]
    # isi dengan temuan otomatis agar selalu ada saran
    for i in insights:
        if i["spec"]:
            specs.append((i["title"], i["detail"], i["spec"]))
    charts, seen = [], set()
    for title, why, spec in specs:
        key = tuple(sorted((k, str(v)) for k, v in spec.items() if v))
        if key in seen:
            continue
        seen.add(key)
        try:
            c = build_chart(df, {**spec, "title": title[:90]})
            c["why"] = why
            charts.append(c)
        except Exception:
            continue
        if len(charts) >= max_charts:
            break
    return {"goal": goal or None, "insights": [{k: v for k, v in i.items() if k != "spec"} for i in insights[:12]], "charts": charts}
