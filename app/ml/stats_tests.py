"""Temuan yang divalidasi secara statistik: uji hipotesis + koreksi Benjamini-Hochberg (FDR) + effect size + mutual information."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from ..utils import infer_kind
from .common import numeric_cols

MAX_NUM, MAX_CAT = 15, 10
MIN_RHO, MIN_ETA2, MIN_V, MIN_MI = 0.30, 0.06, 0.20, 0.30


def bh_fdr(p: list[float]) -> list[float]:
    """Koreksi Benjamini-Hochberg: kembalikan q-value."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


def _fmt_p(p):
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def _cat_cols(df):
    out = []
    for c in df.columns:
        if infer_kind(df[c]) == "categorical" and 2 <= df[c].nunique(dropna=True) <= 30:
            out.append(c)
    return out[:MAX_CAT]


def run_tests(df: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    """Semua uji dijalankan dulu, FDR diterapkan lintas SELURUH uji, baru difilter berdasarkan q-value dan effect size."""
    N = numeric_cols(df)[:MAX_NUM]
    C = _cat_cols(df)
    tests: list[dict] = []

    for a, b in itertools.combinations(N, 2):
        d = df[[a, b]].dropna()
        if len(d) >= 20 and d[a].nunique() > 2 and d[b].nunique() > 2:
            rho, p = stats.spearmanr(d[a], d[b])
            if np.isfinite(rho):
                tests.append({"type": "num_num", "a": a, "b": b, "p": float(p), "effect": float(rho), "n": len(d), "test": "korelasi Spearman"})

    for cc in C:
        for nc in N:
            d = df[[cc, nc]].dropna()
            groups = {k: g[nc].to_numpy() for k, g in d.groupby(cc) if len(g) >= 3}
            if len(groups) < 2 or len(d) < 20 or d[nc].nunique() < 3:
                continue
            arrs = list(groups.values())
            rk = d[nc].rank()  # effect size pada peringkat: tahan outlier dan sejalan dengan uji non-parametrik
            gm = rk.mean()
            sst = float(((rk - gm) ** 2).sum())
            if sst == 0:
                continue
            ssb = float(sum(len(g) * (g.mean() - gm) ** 2 for _, g in rk.groupby(d[cc]) if len(g) >= 3))
            eta2 = ssb / sst
            if len(arrs) == 2:
                p = float(stats.mannwhitneyu(arrs[0], arrs[1], alternative="two-sided").pvalue)
                test = "Mann-Whitney U"
            else:
                p = float(stats.kruskal(*arrs).pvalue)
                test = "Kruskal-Wallis"
            means = {k: float(v.mean()) for k, v in groups.items()}
            tests.append({"type": "cat_num", "a": cc, "b": nc, "p": p, "effect": eta2, "n": len(d), "test": test, "means": means})

    for a, b in itertools.combinations(C, 2):
        d = df[[a, b]].dropna()
        if len(d) < 30:
            continue
        ct = pd.crosstab(d[a], d[b])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        if (exp < 5).mean() > 0.2:
            continue
        v = float(np.sqrt(chi2 / (len(d) * (min(ct.shape) - 1))))
        tests.append({"type": "cat_cat", "a": a, "b": b, "p": float(p), "effect": v, "n": len(d), "test": "chi-square (Cramér's V)"})

    qs = bh_fdr([t["p"] for t in tests])
    for t, q in zip(tests, qs):
        t["q"] = q
    return tests


def significant_insights(df: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    out = []
    tests = run_tests(df, alpha)
    n_tests = len(tests)
    for t in tests:
        if t["q"] >= alpha:
            continue
        note = f"{t['test']}, n={t['n']}, p={_fmt_p(t['p'])}, q(FDR)={_fmt_p(t['q'])}, dari {n_tests} uji"
        if t["type"] == "num_num" and abs(t["effect"]) >= MIN_RHO:
            r = t["effect"]
            kuat = "kuat" if abs(r) >= 0.6 else "sedang"
            out.append({"score": abs(r), "kind": "insight", "title": f"'{t['a']}' dan '{t['b']}' berhubungan {kuat} (ρ = {r:.2f})",
                        "detail": f"Keduanya bergerak {'searah' if r > 0 else 'berlawanan arah'}. Korelasi bukan bukti sebab-akibat.",
                        "spec": {"type": "scatter", "x": t["a"], "y": t["b"]}, "method": note, "p": t["p"], "q": t["q"], "effect": r, "effect_label": "ρ Spearman"})
        elif t["type"] == "cat_num" and t["effect"] >= MIN_ETA2:
            m = sorted(t["means"].items(), key=lambda kv: -kv[1])
            out.append({"score": 0.3 + t["effect"], "kind": "insight", "title": f"'{t['b']}' berbeda nyata antar '{t['a']}'",
                        "detail": f"Rata-rata tertinggi di '{m[0][0]}' ({m[0][1]:.4g}), terendah di '{m[-1][0]}' ({m[-1][1]:.4g}). Kategori menjelaskan ±{t['effect']:.0%} variasi peringkat (η² pada peringkat).",
                        "spec": {"type": "bar", "x": t["a"], "y": t["b"], "agg": "mean"}, "method": note, "p": t["p"], "q": t["q"], "effect": t["effect"], "effect_label": "η² (peringkat)"})
        elif t["type"] == "cat_cat" and t["effect"] >= MIN_V:
            out.append({"score": 0.2 + t["effect"], "kind": "insight", "title": f"'{t['a']}' dan '{t['b']}' saling terkait (Cramér's V = {t['effect']:.2f})",
                        "detail": "Sebaran satu kategori berbeda tergantung kategori lainnya.", "spec": None, "method": note, "p": t["p"], "q": t["q"], "effect": t["effect"], "effect_label": "Cramér's V"})
    out.extend(nonlinear_pairs(df, {(t["a"], t["b"]) for t in tests if t["type"] == "num_num" and t["q"] < alpha and abs(t["effect"]) >= MIN_RHO}))
    out.sort(key=lambda r: -r["score"])
    return out


def nonlinear_pairs(df: pd.DataFrame, skip: set) -> list[dict]:
    """Hubungan yang tak tertangkap korelasi monoton (mis. kurva U) lewat mutual information."""
    N = numeric_cols(df)[:MAX_NUM]
    if len(N) < 2 or len(df) < 50:
        return []
    X = df[N].astype(float)
    X = X.fillna(X.median())
    if len(X) > 5000:
        X = X.sample(5000, random_state=0)
    res = []
    for a in N:
        others = [c for c in N if c != a]
        mi = mutual_info_regression(X[others], X[a], random_state=0)
        for b, m in zip(others, mi):
            if (a, b) in skip or (b, a) in skip or a > b:
                continue
            rho = abs(stats.spearmanr(X[a], X[b])[0])
            h = stats.entropy(np.histogram(X[a], bins=10)[0] + 1e-9)
            nmi = float(m / h) if h > 0 else 0.0
            if nmi >= MIN_MI and rho < MIN_RHO:
                res.append({"score": 0.3 + nmi, "kind": "insight", "title": f"Hubungan non-linear antara '{a}' dan '{b}'",
                            "detail": "Korelasi monoton lemah, tetapi informasi bersama tinggi (mis. pola melengkung). Periksa scatter plot.",
                            "spec": {"type": "scatter", "x": a, "y": b}, "method": f"mutual information ternormalisasi = {nmi:.2f}, |ρ| = {rho:.2f}",
                            "p": None, "q": None, "effect": nmi, "effect_label": "MI/H"})
    return res
