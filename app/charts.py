"""Pembuat diagram: spesifikasi sederhana -> figure Plotly (dict JSON)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import as_datetime, infer_kind, is_numeric, jsonable

CHART_TYPES = ["histogram", "bar", "line", "scatter", "box", "pie", "heatmap"]
AGGS = ["sum", "mean", "median", "min", "max", "count", "nunique"]
MAX_POINTS = 20000
PALETTE = ["#4f7cff", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#64748b"]


def _need(df, *cols):
    for c in cols:
        if c and c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan.")


def _layout(title, xt=None, yt=None, **extra):
    lay = {"title": {"text": title, "font": {"size": 15}}, "xaxis": {"title": xt}, "yaxis": {"title": yt},
           "margin": {"l": 55, "r": 20, "t": 45, "b": 50}, "colorway": PALETTE, "bargap": 0.08}
    lay.update(extra)
    return lay


def _agg_series(df, x, y, agg, top=20):
    if not y or agg == "count":
        g = df.groupby(x, dropna=True).size()
        label = "jumlah"
    else:
        if agg not in ("count", "nunique") and not is_numeric(df[y]):
            raise ValueError(f"Kolom '{y}' bukan numerik, tidak bisa di-{agg}. Pilih agregasi 'count'.")
        g = df.groupby(x, dropna=True)[y].agg(agg)
        label = f"{agg}({y})"
    return g, label


def build_chart(df: pd.DataFrame, spec: dict) -> dict:
    t = spec.get("type")
    if t not in CHART_TYPES:
        raise ValueError(f"Jenis diagram '{t}' tidak dikenal. Pilihan: {', '.join(CHART_TYPES)}")
    x, y, color = spec.get("x") or None, spec.get("y") or None, spec.get("color") or None
    agg = spec.get("agg") or "sum"
    if agg not in AGGS:
        raise ValueError(f"Agregasi '{agg}' tidak dikenal.")
    _need(df, x, y, color)

    if t == "histogram":
        if not x:
            raise ValueError("Histogram membutuhkan kolom X.")
        s = df[x].dropna()
        if is_numeric(df[x]):
            s = s.sample(MAX_POINTS, random_state=0) if len(s) > MAX_POINTS else s
            fig = {"data": [{"type": "histogram", "x": s.tolist(), "nbinsx": int(spec.get("bins") or 0) or None, "marker": {"color": PALETTE[0]}}],
                   "layout": _layout(spec.get("title") or f"Distribusi {x}", x, "jumlah")}
        else:
            vc = s.astype(str).value_counts().head(25)
            fig = {"data": [{"type": "bar", "x": vc.index.tolist(), "y": vc.values.tolist(), "marker": {"color": PALETTE[0]}}],
                   "layout": _layout(spec.get("title") or f"Frekuensi {x}", x, "jumlah")}

    elif t == "bar":
        if not x:
            raise ValueError("Diagram batang membutuhkan kolom X (kategori).")
        if is_numeric(df[x]) and df[x].nunique() > 30:
            raise ValueError(f"'{x}' numerik dengan banyak nilai unik; pakai histogram atau pilih kolom kategori.")
        g, label = _agg_series(df, x, y, agg)
        g = g.sort_values(ascending=False).head(20)
        fig = {"data": [{"type": "bar", "x": [str(i) for i in g.index], "y": g.values.tolist(), "marker": {"color": PALETTE[0]}}],
               "layout": _layout(spec.get("title") or f"{label} per {x}", x, label)}

    elif t == "line":
        if not x or not y:
            raise ValueError("Diagram garis membutuhkan kolom X dan Y.")
        if not is_numeric(df[y]) and agg not in ("count", "nunique"):
            raise ValueError(f"Kolom Y '{y}' harus numerik.")
        d = df[[x, y]].dropna()
        if infer_kind(df[x]) == "datetime":
            d = d.assign(**{x: as_datetime(d[x])}).dropna().set_index(x).sort_index()
            span = (d.index.max() - d.index.min()).days if len(d) else 0
            freq = "D" if span <= 90 else "MS" if span <= 1100 else "YS"
            g = d[y].resample(freq).agg(agg if agg != "count" else "count")
            xs = [i.isoformat() for i in g.index]
        else:
            g = d.groupby(x)[y].agg(agg).sort_index()
            xs = [jsonable(i) for i in g.index]
        label = f"{agg}({y})"
        fig = {"data": [{"type": "scatter", "mode": "lines+markers", "x": xs, "y": g.values.tolist(), "line": {"color": PALETTE[0]}}],
               "layout": _layout(spec.get("title") or f"{label} menurut {x}", x, label)}

    elif t == "scatter":
        if not x or not y:
            raise ValueError("Scatter membutuhkan kolom X dan Y.")
        cols = [x, y] + ([color] if color else [])
        d = df[cols].dropna()
        if len(d) > MAX_POINTS:
            d = d.sample(MAX_POINTS, random_state=0)
        if color and not is_numeric(df[color]):
            top = d[color].astype(str).value_counts().head(10).index
            traces = [{"type": "scattergl", "mode": "markers", "name": str(k), "x": g[x].tolist(), "y": g[y].tolist(), "marker": {"size": 6, "opacity": 0.7}}
                      for k, g in d[d[color].astype(str).isin(top)].groupby(d[color].astype(str))]
        else:
            m = {"size": 6, "opacity": 0.7, "color": PALETTE[0]}
            if color:
                m.update({"color": d[color].tolist(), "colorscale": "Viridis", "showscale": True})
            traces = [{"type": "scattergl", "mode": "markers", "x": d[x].tolist(), "y": d[y].tolist(), "marker": m}]
        fig = {"data": traces, "layout": _layout(spec.get("title") or f"{y} vs {x}", x, y)}

    elif t == "box":
        if not y or not is_numeric(df[y]):
            raise ValueError("Box plot membutuhkan kolom Y numerik.")
        if x:
            top = df[x].astype(str).value_counts().head(12).index
            traces = [{"type": "box", "name": str(k), "y": g[y].dropna().tolist()} for k, g in df[df[x].astype(str).isin(top)].groupby(df[x].astype(str))]
        else:
            traces = [{"type": "box", "name": y, "y": df[y].dropna().tolist(), "marker": {"color": PALETTE[0]}}]
        fig = {"data": traces, "layout": _layout(spec.get("title") or (f"{y} per {x}" if x else f"Sebaran {y}"), x, y, showlegend=False)}

    elif t == "pie":
        if not x:
            raise ValueError("Pie membutuhkan kolom X (kategori).")
        g, label = _agg_series(df, x, y, agg)
        g = g[g > 0].sort_values(ascending=False)
        if len(g) > 8:
            g = pd.concat([g.head(7), pd.Series({"Lainnya": g.iloc[7:].sum()})])
        fig = {"data": [{"type": "pie", "labels": [str(i) for i in g.index], "values": g.values.tolist(), "hole": 0.4}],
               "layout": _layout(spec.get("title") or f"Komposisi {x}")}

    else:  # heatmap korelasi
        num = df.select_dtypes(include="number").loc[:, lambda d: d.nunique() > 1].iloc[:, :25]
        if num.shape[1] < 2:
            raise ValueError("Heatmap korelasi membutuhkan minimal 2 kolom numerik.")
        c = num.corr()
        fig = {"data": [{"type": "heatmap", "z": jsonable(c.values), "x": list(c.columns), "y": list(c.columns), "zmin": -1, "zmax": 1,
                         "colorscale": "RdBu", "reversescale": True}],
               "layout": _layout(spec.get("title") or "Korelasi antar kolom numerik", margin={"l": 110, "r": 20, "t": 45, "b": 110})}

    return {"title": fig["layout"]["title"]["text"], "figure": jsonable(fig), "spec": {k: v for k, v in spec.items() if v not in (None, "")}}


def auto_charts(df: pd.DataFrame, limit: int = 8) -> list[dict]:
    """Kumpulan diagram otomatis untuk gambaran awal data."""
    out, kinds = [], {c: infer_kind(df[c]) for c in df.columns}
    nums = [c for c, k in kinds.items() if k == "numeric"]
    cats = [c for c, k in kinds.items() if k == "categorical" and 1 < df[c].nunique() <= 20]
    dts = [c for c, k in kinds.items() if k == "datetime"]

    def add(spec):
        try:
            out.append(build_chart(df, spec))
        except Exception:
            pass

    for c in nums[:3]:
        add({"type": "histogram", "x": c})
    for c in cats[:2]:
        add({"type": "bar", "x": c, "agg": "count"})
    if dts and nums:
        add({"type": "line", "x": dts[0], "y": nums[0], "agg": "sum"})
    if len(nums) >= 2:
        add({"type": "heatmap"})
    return out[:limit]
