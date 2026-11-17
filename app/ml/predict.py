"""AutoML ringan: bandingkan baseline vs model, validasi silang, pentingnya fitur (permutation), deteksi kebocoran data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utils import as_datetime, infer_kind, is_numeric
from .common import layout, need, result, table

MAX_ROWS = 50000


def _features(df: pd.DataFrame, target: str, features):
    X = pd.DataFrame(index=df.index)
    dropped, leaks = [], []
    y = df[target]
    for c in (features or [c for c in df.columns if c != target]):
        if c == target or c not in df.columns:
            continue
        s = df[c]
        k = infer_kind(s)
        nun = s.nunique(dropna=True)
        if nun <= 1:
            dropped.append(f"{c} (konstan)")
        elif k == "datetime":
            d = as_datetime(s)
            X[f"{c}__tahun"], X[f"{c}__bulan"], X[f"{c}__hari_minggu"] = d.dt.year, d.dt.month, d.dt.dayofweek
        elif k == "text" or (k != "numeric" and nun > 30):
            dropped.append(f"{c} (teks/kardinalitas tinggi)")
        elif k == "numeric" and nun / max(s.notna().sum(), 1) > 0.98 and pd.api.types.is_integer_dtype(s) and (s.dropna().diff().dropna() > 0).mean() > .99:
            dropped.append(f"{c} (tampak seperti ID/nomor urut)")
        else:
            X[c] = s.astype(float) if k in ("numeric", "boolean") and is_numeric(s) else s.astype(str).where(s.notna())
    return X, y, dropped


def _leakage(X: pd.DataFrame, y: pd.Series, task: str) -> list[str]:
    out = []
    for c in X.columns:
        d = pd.DataFrame({"x": X[c], "y": y}).dropna()
        if len(d) < 30:
            continue
        if is_numeric(d["x"]) and task == "regression":
            r = abs(d["x"].corr(d["y"], method="spearman")) if d["x"].nunique() > 1 and d["y"].nunique() > 1 else 0.0
            if r >= 0.98:
                out.append(f"'{c}' berkorelasi {r:.3f} dengan target (hampir identik): kemungkinan kebocoran data.")
        elif task == "classification":
            acc = d.groupby("x")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(d) if d["x"].nunique() < len(d) * 0.5 else 0
            if acc >= 0.99 and d["y"].value_counts(normalize=True).iloc[0] < 0.9:
                out.append(f"'{c}' menentukan target hampir sempurna (akurasi {acc:.1%} sendirian): kemungkinan kebocoran data.")
    return out


def _preprocessor(X, scale):
    num = [c for c in X.columns if is_numeric(X[c])]
    cat = [c for c in X.columns if c not in num]
    steps = []
    if num:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True))] + ([("sc", StandardScaler())] if scale else [])), num))
    if cat:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), cat))
    return ColumnTransformer(steps, sparse_threshold=0)


def automl(df: pd.DataFrame, target: str, features=None) -> dict:
    need(target in df.columns, f"Kolom target '{target}' tidak ditemukan.")
    d = df[df[target].notna()]
    need(len(d) >= 40, "Butuh minimal 40 baris dengan target terisi.")
    if len(d) > MAX_ROWS:
        d = d.sample(MAX_ROWS, random_state=0)
    X, y, dropped = _features(d, target, features)
    need(X.shape[1] >= 1, "Tidak ada fitur yang bisa dipakai.")
    task = "regression" if is_numeric(d[target]) and d[target].nunique() > 15 else "classification"
    if task == "classification":
        y = y.astype(str)
        vc = y.value_counts()
        need(len(vc) >= 2, "Target hanya punya satu kelas.")
        need(len(vc) <= 30, "Target punya terlalu banyak kelas (>30).")
        keep = vc[vc >= 5].index
        need(len(keep) >= 2, "Setiap kelas butuh minimal 5 baris.")
        m = y.isin(keep)
        X, y = X[m], y[m]
    leaks = _leakage(X, y, task)
    models = {
        "Baseline (rata-rata/mayoritas)": (DummyRegressor() if task == "regression" else DummyClassifier(strategy="most_frequent"), False),
        "Linear (Ridge/Logistic)": (Ridge(alpha=1.0) if task == "regression" else LogisticRegression(max_iter=1000), True),
        "Gradient Boosting": (HistGradientBoostingRegressor(random_state=0) if task == "regression" else HistGradientBoostingClassifier(random_state=0), False),
    }
    folds = 5 if len(X) >= 100 else 3
    cv = KFold(folds, shuffle=True, random_state=0) if task == "regression" else StratifiedKFold(min(folds, int(y.value_counts().min())), shuffle=True, random_state=0)
    scoring = "r2" if task == "regression" else "f1_macro"
    rows, fitted = [], {}
    for name, (est, scale) in models.items():
        pipe = Pipeline([("pre", _preprocessor(X, scale)), ("m", est)])
        sc = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        rows.append({"model": name, "skor_rata2": round(float(sc.mean()), 4), "simpangan": round(float(sc.std()), 4)})
        fitted[name] = pipe
    best = max((r for r in rows if not r["model"].startswith("Baseline")), key=lambda r: r["skor_rata2"])
    base = next(r for r in rows if r["model"].startswith("Baseline"))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y if task == "classification" else None)
    pipe = fitted[best["model"]].fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    holdout = float(r2_score(yte, pred)) if task == "regression" else float(f1_score(yte, pred, average="macro"))
    pi = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=0, scoring=scoring)
    imp = pd.DataFrame({"fitur": X.columns, "penting": pi.importances_mean, "sd": pi.importances_std}).sort_values("penting", ascending=False)
    top = imp.head(12).iloc[::-1]
    metric = "R²" if task == "regression" else "F1 makro"
    warnings = list(dict.fromkeys(f"Dikecualikan: {x}" for x in dropped)) + leaks
    if best["skor_rata2"] - base["skor_rata2"] < 0.05:
        warnings.append("Model nyaris tidak lebih baik dari baseline: fitur yang ada mungkin kurang menjelaskan target.")
    if best["skor_rata2"] - holdout > 0.15:
        warnings.append("Skor holdout jauh di bawah validasi silang: kemungkinan overfitting.")
    if task == "regression" and abs(float(y.skew())) > 2:
        warnings.append(f"Target sangat miring (skew {float(y.skew()):.1f}) / punya outlier ekstrem: R² bisa menyesatkan. Bersihkan outlier atau pertimbangkan transformasi log di tab Transformasi.")
    if len(X) < 200:
        warnings.append("Data kecil (<200 baris): skor kurang stabil.")
    charts = [{"title": "Pentingnya fitur", "figure": {"data": [{"type": "bar", "orientation": "h", "x": top["penting"].tolist(), "y": top["fitur"].tolist(),
              "error_x": {"type": "data", "array": top["sd"].tolist()}, "marker": {"color": "#4f7cff"}}], "layout": layout(f"Permutation importance (turunnya {metric} bila fitur diacak)", metric, None, margin={"l": 140, "r": 20, "t": 44, "b": 50})}}]
    if task == "regression":
        lim = [float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))]
        charts.append({"title": "Aktual vs prediksi", "figure": {"data": [{"type": "scattergl", "mode": "markers", "x": yte.tolist(), "y": pred.tolist(), "marker": {"size": 5, "opacity": .6}, "name": "data uji"},
                       {"type": "scatter", "mode": "lines", "x": lim, "y": lim, "line": {"dash": "dash", "color": "#64748b"}, "name": "ideal"}], "layout": layout("Aktual vs prediksi (data uji)", "aktual", "prediksi")}})
    else:
        labs = sorted(y.unique())
        cm = confusion_matrix(yte, pred, labels=labs)
        charts.append({"title": "Matriks konfusi", "figure": {"data": [{"type": "heatmap", "z": cm.tolist(), "x": labs, "y": labs, "colorscale": "Blues"}], "layout": layout("Matriks konfusi (data uji)", "prediksi", "aktual")}})
    top3 = ", ".join(f"{r.fitur}" for r in imp.head(3).itertuples())
    head_ = f"Tugas: {'regresi' if task == 'regression' else 'klasifikasi'} untuk '{target}' ({len(X)} baris, {X.shape[1]} fitur). "
    if best["skor_rata2"] <= base["skor_rata2"] + 0.02:
        summary = head_ + f"Tidak ada model yang mengalahkan baseline ({metric} CV terbaik {best['skor_rata2']:.3f} vs baseline {base['skor_rata2']:.3f}); fitur yang ada tidak cukup menjelaskan target, jadi urutan fitur di bawah tidak boleh diandalkan."
    else:
        summary = head_ + f"Model terbaik: {best['model']} ({metric} CV {best['skor_rata2']:.3f} vs baseline {base['skor_rata2']:.3f}; holdout {holdout:.3f}). Fitur paling berpengaruh: {top3}."
    return result(f"Prediksi '{target}'", "AutoML ringan (CV + permutation importance)", summary,
                  assumptions=[f"Validasi silang {folds}-lipat; skor = {metric}.", "Kategori kecil (<5 baris) digabung; kolom ID/teks panjang dikecualikan; nilai kosong diisi median/modus di dalam pipeline (tanpa bocor ke data uji).",
                               "Pentingnya fitur menunjukkan kontribusi prediktif, bukan sebab-akibat."],
                  warnings=warnings, tables=[table("Perbandingan model", pd.DataFrame(rows)), table("Pentingnya fitur", imp.round(4))], charts=charts,
                  stats={"task": task, "best_model": best["model"], "cv_score": best["skor_rata2"], "baseline": base["skor_rata2"], "holdout": holdout, "top_features": imp.head(5)["fitur"].tolist(), "metric": metric})
