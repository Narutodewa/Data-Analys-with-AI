"""Analisis deret waktu: dekomposisi musiman (STL), titik perubahan, peramalan dengan backtest jujur."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from ..utils import as_datetime
from .common import layout, need, result, table

FREQ = {"D": ("hari", 7), "W": ("minggu", 52), "MS": ("bulan", 12)}


def _prepare(df, date_col, value_col, agg):
    need(date_col in df.columns and value_col in df.columns, "Kolom tanggal/nilai tidak ditemukan.")
    d = pd.DataFrame({"t": as_datetime(df[date_col]), "v": pd.to_numeric(df[value_col], errors="coerce")}).dropna()
    need(len(d) >= 12, "Butuh minimal 12 titik data bertanggal.")
    span = (d["t"].max() - d["t"].min()).days
    freq = "D" if span <= 120 else "W" if span <= 900 else "MS"
    agg = agg if agg in ("sum", "mean", "median", "min", "max", "count") else "sum"
    ds_ = d.set_index("t")["v"].sort_index()
    s = ds_.resample(freq).agg(agg)
    cnt = ds_.resample(freq).count()
    trimmed = []
    if agg in ("sum", "count") and len(s) > 14:  # periode awal/akhir yang tidak lengkap menyesatkan total
        med = float(cnt.median())
        if cnt.iloc[-1] < 0.6 * med:
            s, cnt = s.iloc[:-1], cnt.iloc[:-1]
            trimmed.append("terakhir")
        if cnt.iloc[0] < 0.6 * med:
            s, cnt = s.iloc[1:], cnt.iloc[1:]
            trimmed.append("pertama")
    n_gap = int(s.isna().sum())
    s = s.interpolate(limit_direction="both") if agg != "sum" else s.fillna(0.0)
    need(len(s) >= 12, "Setelah diagregasi terlalu sedikit periode (<12).")
    return s.astype(float), freq, n_gap, trimmed


def changepoints(y: np.ndarray, min_size: int = 5, max_cp: int = 4) -> list[int]:
    """Segmentasi biner (perubahan rata-rata) dengan penalti BIC sederhana. Tanpa dependensi tambahan."""
    n = len(y)
    var = np.var(np.diff(y)) / 2 if n > 2 else np.var(y)
    pen = 2 * np.log(n) * max(var, 1e-12)

    def sse(a, b):
        seg = y[a:b]
        return float(((seg - seg.mean()) ** 2).sum())

    cps: list[int] = []
    segs = [(0, n)]
    while segs and len(cps) < max_cp:
        best = None
        for a, b in segs:
            if b - a < 2 * min_size:
                continue
            base = sse(a, b)
            for k in range(a + min_size, b - min_size + 1):
                gain = base - sse(a, k) - sse(k, b)
                if best is None or gain > best[0]:
                    best = (gain, k, a, b)
        if not best or best[0] < pen:
            break
        _, k, a, b = best
        cps.append(k)
        segs.remove((a, b))
        segs += [(a, k), (k, b)]
    return sorted(cps)


def _fit(y: pd.Series, period: int | None):
    seasonal = "add" if period and len(y) >= 2 * period else None
    try:
        return ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=seasonal, seasonal_periods=period if seasonal else None,
                                    initialization_method="estimated").fit(optimized=True)
    except Exception:
        return ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()


def _smape(a, f):
    a, f = np.asarray(a), np.asarray(f)
    den = (np.abs(a) + np.abs(f)) / 2
    return float(np.mean(np.where(den == 0, 0, np.abs(a - f) / np.where(den == 0, 1, den))) * 100)


def analyze(df, date_col, value_col, agg="sum", horizon=None) -> dict:
    s, freq, n_gap, trimmed = _prepare(df, date_col, value_col, agg)
    unit, period = FREQ[freq]
    n = len(s)
    horizon = int(horizon or max(3, min(period if freq != "W" else 8, n // 4)))
    horizon = max(1, min(horizon, max(1, n // 2)))
    warnings, assumptions = [], [f"Data diagregasi per {unit} ({agg}); {n} periode."]
    if trimmed:
        assumptions.append(f"Periode {' dan '.join(trimmed)} dibuang karena datanya tidak lengkap (agregasi {agg} akan menyesatkan).")
    if n_gap:
        assumptions.append(f"{n_gap} periode tanpa data diisi {'0' if agg == 'sum' else 'interpolasi'}.")

    seasonal_strength = None
    has_season = n >= 2 * period and period > 1
    trend_line = s.rolling(max(3, period if has_season else max(3, n // 10)), center=True, min_periods=1).mean()
    seasonal_part = None
    if has_season:
        stl = STL(s, period=period, robust=True).fit()
        trend_line = stl.trend
        seasonal_part = stl.seasonal
        var_r = float(np.var(stl.resid))
        seasonal_strength = max(0.0, 1 - var_r / float(np.var(stl.seasonal + stl.resid)))
        assumptions.append(f"Dekomposisi STL robust dengan periode {period}.")
    else:
        assumptions.append("Data terlalu pendek untuk menilai musiman; hanya tren yang dihitung.")

    cps = changepoints(s.to_numpy())
    cp_dates = [s.index[k] for k in cps]

    # backtest jujur: bandingkan dengan baseline naif
    h_bt = max(2, min(horizon, n // 5))
    train, test = s.iloc[:-h_bt], s.iloc[-h_bt:]
    bt = _fit(train, period if freq != "W" else (period if n >= 120 else None)).forecast(h_bt)
    naive = (np.repeat(train.iloc[-period], h_bt) if has_season and len(train) >= period else np.repeat(train.iloc[-1], h_bt))
    e_model, e_naive = _smape(test, bt), _smape(test, naive)
    beats = e_model < e_naive
    if not beats:
        warnings.append(f"Model tidak mengalahkan baseline naif pada backtest (sMAPE {e_model:.1f}% vs {e_naive:.1f}%). Perlakukan ramalan sebagai indikatif saja.")
    if n < 24:
        warnings.append("Data pendek (<24 periode): interval ramalan kurang andal.")

    model = _fit(s, period if has_season else None)
    fc = model.forecast(horizon)
    resid_sd = float(np.std(s - model.fittedvalues)) or 1e-9
    steps = np.arange(1, horizon + 1)
    band = 1.96 * resid_sd * np.sqrt(steps)
    fidx = fc.index

    traces = [
        {"type": "scatter", "mode": "lines", "name": "aktual", "x": [i.isoformat() for i in s.index], "y": s.tolist(), "line": {"color": "#4f7cff"}},
        {"type": "scatter", "mode": "lines", "name": "tren", "x": [i.isoformat() for i in s.index], "y": trend_line.tolist(), "line": {"color": "#f59e0b", "dash": "dot"}},
        {"type": "scatter", "mode": "lines", "name": "ramalan", "x": [i.isoformat() for i in fidx], "y": fc.tolist(), "line": {"color": "#10b981"}},
        {"type": "scatter", "mode": "lines", "name": "interval 95%", "x": [i.isoformat() for i in fidx] + [i.isoformat() for i in fidx][::-1],
         "y": (fc + band).tolist() + (fc - band).tolist()[::-1], "fill": "toself", "fillcolor": "rgba(16,185,129,.18)", "line": {"width": 0}, "hoverinfo": "skip"},
    ]
    shapes = [{"type": "line", "x0": d.isoformat(), "x1": d.isoformat(), "yref": "paper", "y0": 0, "y1": 1, "line": {"color": "#ef4444", "dash": "dash", "width": 1}} for d in cp_dates]
    charts = [{"title": "Tren dan ramalan", "figure": {"data": traces, "layout": layout(f"{value_col} per {unit}", date_col, value_col, shapes=shapes)}}]
    if seasonal_part is not None:
        pos = pd.Series(seasonal_part.to_numpy()).groupby(np.arange(n) % period).mean()
        charts.append({"title": "Pola musiman", "figure": {"data": [{"type": "bar", "x": [int(i) + 1 for i in pos.index], "y": pos.tolist(), "marker": {"color": "#8b5cf6"}}],
                                                              "layout": layout(f"Pola musiman (periode {period} {unit})", f"posisi dalam siklus", "efek")}})
    w = max(3, n // 10)
    tr0, tr1 = float(s.iloc[:w].median()), float(s.iloc[-w:].median())  # median data aktual: tahan outlier
    chg = (tr1 - tr0) / abs(tr0) * 100 if tr0 else float("nan")
    rho, pv = stats.spearmanr(np.arange(n), s.to_numpy())
    if pv < 0.05 and abs(rho) >= 0.2:
        parts = [f"Tren {'naik' if rho > 0 else 'turun'} yang signifikan (ρ = {rho:.2f}, p {'<0,001' if pv < 0.001 else '= ' + format(pv, '.3f')})" + (f"; ±{abs(chg):.0f}% dari awal ke akhir (median 10% awal vs akhir)." if np.isfinite(chg) else ".")]
    else:
        parts = [f"Tidak ada tren monoton yang signifikan (ρ = {rho:.2f}, p = {pv:.2f})" + (f"; selisih awal-akhir ±{abs(chg):.0f}% kemungkinan fluktuasi biasa." if np.isfinite(chg) else ".")]
    if seasonal_strength is not None:
        parts.append(f"Kekuatan musiman {seasonal_strength:.0%}" + (" (kuat)." if seasonal_strength > .6 else " (lemah)." if seasonal_strength < .2 else " (sedang)."))
    if cp_dates:
        parts.append("Perubahan level terdeteksi sekitar " + ", ".join(d.strftime("%Y-%m-%d") for d in cp_dates) + ".")
    parts.append(f"Ramalan {horizon} {unit} ke depan: {fc.iloc[0]:.4g} → {fc.iloc[-1]:.4g}.")
    tbl = pd.DataFrame({"periode": [i.strftime("%Y-%m-%d") for i in fidx], "ramalan": fc.round(3).to_numpy(), "bawah_95": (fc - band).round(3).to_numpy(), "atas_95": (fc + band).round(3).to_numpy()})
    return result("Deret waktu: tren, musiman, perubahan, ramalan", "STL + Holt-Winters (damped) + segmentasi biner", " ".join(parts),
                  assumptions=assumptions + [f"Interval ≈ ramalan ± 1,96·σ_residu·√h (aproksimasi)."], warnings=warnings,
                  tables=[table("Ramalan", tbl)], charts=charts,
                  stats={"periods": n, "freq": freq, "changepoints": [d.isoformat() for d in cp_dates], "seasonal_strength": seasonal_strength,
                         "backtest_smape_model": e_model, "backtest_smape_naive": e_naive, "beats_naive": bool(beats), "horizon": horizon})
