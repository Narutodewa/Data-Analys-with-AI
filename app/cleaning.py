"""Deteksi masalah data + aksi pembersihan yang dipilih pengguna (hapus / isi rata-rata / biarkan / edit manual)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .store import Dataset
from .utils import infer_kind, is_numeric, jsonable

try:  # lapisan ML opsional: tanpa scikit-learn semua fitur dasar tetap jalan
    from .ml import fuzzy as _fuzzy, impute as _impute, outliers as _outliers, pii as _pii
    HAS_ML = True
except ImportError:  # pragma: no cover
    HAS_ML = False

MAX_ROWS_LISTED = 200
_CURRENCY = r"[\s$€£%]|Rp\.?|IDR|USD"


# ------------------------------------------------------------------ helper
def _to_number(s: pd.Series) -> pd.Series:
    """Parse angka dari teks seperti '$1,200' atau 'Rp 5000'. (Pemisah ribuan titik gaya ID tidak ditebak.)"""
    t = s.astype("string").str.replace(_CURRENCY, "", regex=True).str.replace(",", "", regex=False)
    return pd.to_numeric(t, errors="coerce")


def _is_stringy(s: pd.Series) -> bool:
    return not is_numeric(s) and not pd.api.types.is_bool_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)


def _canon(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.lower()


def _iqr_bounds(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _rows(idx) -> list:
    return [jsonable(i) for i in list(idx)[:MAX_ROWS_LISTED]]


def _sample(df: pd.DataFrame, idx, col, k=5) -> list[dict]:
    out = []
    for i in list(idx)[:k]:
        out.append({"row": jsonable(i), "value": jsonable(df.at[i, col]) if col in df.columns else None})
    return out


# ---------------------------------------------------------------- deteksi
def _numeric_stats(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    return {"mean": jsonable(round(float(s.mean()), 4)), "median": jsonable(float(s.median()))}


def detect_issues(df: pd.DataFrame, ignored: set | None = None) -> list[dict]:
    ignored = ignored or set()
    n = len(df)
    issues: list[dict] = []
    if n == 0:
        return issues

    # 1) baris duplikat
    dup = df.duplicated(keep="first")
    if dup.any():
        idx = df.index[dup]
        issues.append({
            "id": "duplicates", "type": "duplicates", "column": None, "severity": "medium",
            "title": f"{int(dup.sum())} baris duplikat",
            "detail": "Baris identik dengan baris sebelumnya. Duplikat bisa menggandakan hitungan dan merusak rata-rata.",
            "count": int(dup.sum()), "pct": round(100 * dup.mean(), 1), "rows": _rows(idx),
            "samples": [], "stats": {},
            "actions": [{"key": "drop_rows", "label": "Hapus duplikat (simpan yang pertama)"}, {"key": "keep", "label": "Biarkan"}],
            "recommended": "drop_rows",
        })

    for col in df.columns:
        s = df[col]
        na = s.isna()
        nun = s.nunique(dropna=True)

        # 1b) data pribadi (PII)
        if HAS_ML:
            f = _pii.scan_column(s, str(col)) if s.dtype.kind not in "fb" else None
            if f:
                issues.append({
                    "id": f"pii:{col}", "type": "pii", "column": col, "severity": "high",
                    "title": f"Kolom '{col}' diduga berisi data pribadi ({f['type']})",
                    "detail": f"Terdeteksi lewat {f['via']}" + (f" ({f['share']:.0%} nilai cocok)" if f["share"] else "") + f". Contoh tersamarkan: {f['example']}. Samarkan sebelum data dibagikan.",
                    "count": int(s.notna().sum()), "pct": 100.0, "rows": [], "samples": [], "stats": {"jenis": f["type"]},
                    "actions": [{"key": "mask", "label": "Samarkan (a***@x.com)"}, {"key": "hash", "label": "Pseudonim (hash)"}, {"key": "drop_column", "label": "Hapus kolom"}, {"key": "keep", "label": "Biarkan"}],
                    "recommended": "keep",
                })

        # 2) nilai kosong
        if na.any():
            cnt, pct = int(na.sum()), round(100 * na.mean(), 1)
            numeric = is_numeric(s)
            stats = _numeric_stats(s) if numeric else {}
            actions = [{"key": "drop_rows", "label": f"Hapus {cnt} baris"}]
            if numeric:
                actions += [
                    {"key": "fill_mean", "label": f"Isi rata-rata ({stats.get('mean')})" if stats else "Isi rata-rata"},
                    {"key": "fill_median", "label": f"Isi median ({stats.get('median')})" if stats else "Isi median"},
                ]
            mode = s.mode(dropna=True)
            if len(mode):
                stats["mode"] = jsonable(mode.iloc[0])
                actions.append({"key": "fill_mode", "label": f"Isi modus ({stats['mode']})"})
            model_ok = HAS_ML and numeric and s.notna().sum() >= 10 and len(_impute.numeric_cols(df.drop(columns=[col]))) >= 1
            if model_ok:
                actions += [{"key": "fill_knn", "label": "Isi KNN (dari baris mirip)"}, {"key": "fill_iterative", "label": "Isi iteratif/MICE (regresi antar kolom)"}]
            actions += [
                {"key": "fill_value", "label": "Isi nilai tertentu…", "needs_value": True},
                {"key": "drop_column", "label": "Hapus kolom"},
                {"key": "keep", "label": "Biarkan"},
            ]
            note = None
            if model_ok:
                try:
                    note = _impute.missingness_note(df, col)
                except Exception:
                    note = None
            if pct >= 60:
                rec = "drop_column"
            elif numeric:
                clean = s.dropna()
                skew = abs(float(clean.skew())) if len(clean) > 2 else 0
                rec = "fill_median" if skew > 1 else "fill_mean"
            elif len(mode):
                rec = "fill_mode"
            else:
                rec = "keep"
            if note and "TIDAK acak" in note and pct < 60:
                rec = "fill_knn"
            issues.append({
                "id": f"missing:{col}", "type": "missing", "column": col,
                "severity": "high" if pct >= 30 else "medium" if pct >= 5 else "low",
                "title": f"Kolom '{col}': {cnt} nilai kosong ({pct}%)",
                "detail": "Pilih: hapus baris, isi dengan statistik/model, isi manual, atau biarkan." + (f" {note}" if note else ""),
                "count": cnt, "pct": pct, "rows": _rows(df.index[na]), "samples": [],
                "stats": stats, "actions": actions, "recommended": rec,
            })

        # 3) angka tersimpan sebagai teks / campuran tipe
        if _is_stringy(s) and s.notna().sum() >= 3:
            nn = s.notna()
            plain = pd.to_numeric(s, errors="coerce")
            cleaned = _to_number(s)
            share = cleaned[nn].notna().mean()
            if share >= 0.8:
                bad = nn & cleaned.isna()
                fixable = nn & plain.isna() & cleaned.notna()
                if bad.any() or fixable.any():
                    badn, fixn = int(bad.sum()), int(fixable.sum())
                    idx = df.index[bad | fixable]
                    issues.append({
                        "id": f"mixed_type:{col}", "type": "mixed_type", "column": col,
                        "severity": "high" if badn else "medium",
                        "title": f"Kolom '{col}' seharusnya angka, tapi {badn + fixn} nilai bermasalah",
                        "detail": f"{fixn} nilai bisa dibersihkan (simbol mata uang/koma), {badn} nilai bukan angka sama sekali.",
                        "count": badn + fixn, "pct": round(100 * (badn + fixn) / n, 1), "rows": _rows(idx),
                        "samples": _sample(df, idx, col), "stats": {},
                        "actions": [
                            {"key": "convert_numeric", "label": "Ubah ke angka (yang tak valid → kosong)"},
                            {"key": "drop_rows", "label": f"Hapus baris tak valid ({badn})"},
                            {"key": "keep", "label": "Biarkan"},
                        ],
                        "recommended": "convert_numeric",
                    })
                    continue  # jangan lanjut cek teks/outlier untuk kolom ini

        # 4) outlier (IQR)
        if is_numeric(s) and s.notna().sum() >= 8 and nun > 5:
            lo, hi = _iqr_bounds(s.dropna())
            mask = (s < lo) | (s > hi)
            if mask.any():
                cnt = int(mask.sum())
                issues.append({
                    "id": f"outliers:{col}", "type": "outliers", "column": col,
                    "severity": "medium" if cnt / n > 0.02 else "low",
                    "title": f"Kolom '{col}': {cnt} outlier",
                    "detail": f"Di luar rentang wajar [{lo:.4g}, {hi:.4g}] (metode IQR). Bisa data salah input atau memang kasus ekstrem.",
                    "count": cnt, "pct": round(100 * cnt / n, 1), "rows": _rows(df.index[mask]),
                    "samples": _sample(df, df.index[mask], col),
                    "stats": {"lower": jsonable(float(lo)), "upper": jsonable(float(hi)), **_numeric_stats(s)},
                    "actions": [
                        {"key": "drop_rows", "label": f"Hapus {cnt} baris"},
                        {"key": "clip", "label": "Batasi ke rentang wajar (winsorize)"},
                        {"key": "fill_median", "label": "Ganti dengan median"},
                        {"key": "fill_value", "label": "Ganti nilai tertentu…", "needs_value": True},
                        {"key": "keep", "label": "Biarkan (valid)"},
                    ],
                    "recommended": "clip",
                })

        # 5) teks tidak konsisten (spasi / kapitalisasi)
        if _is_stringy(s) and 1 < nun <= max(50, int(0.5 * s.notna().sum())) and infer_kind(s) == "categorical":
            nn = s.notna()
            canon = _canon(s)
            raw = s.astype("string")
            groups = pd.DataFrame({"raw": raw[nn], "canon": canon[nn]})
            variants = groups.groupby("canon")["raw"].nunique()
            multi = variants[variants > 1].index
            if len(multi):
                mapping = {}
                for c in multi:
                    vc = groups.loc[groups["canon"] == c, "raw"].value_counts()
                    top = vc.index[0]
                    for v in vc.index[1:]:
                        mapping[v] = top
                affected = raw.isin(list(mapping.keys()))
                issues.append({
                    "id": f"text:{col}", "type": "inconsistent_text", "column": col, "severity": "low",
                    "title": f"Kolom '{col}': penulisan tidak konsisten ({int(affected.sum())} nilai)",
                    "detail": "Variasi spasi/huruf besar-kecil: " + ", ".join(f"'{k}'→'{v}'" for k, v in list(mapping.items())[:4]),
                    "count": int(affected.sum()), "pct": round(100 * affected.mean(), 1),
                    "rows": _rows(df.index[affected]), "samples": _sample(df, df.index[affected], col),
                    "stats": {"mapping": mapping},
                    "actions": [{"key": "normalize", "label": "Seragamkan (pakai variasi terbanyak)"}, {"key": "keep", "label": "Biarkan"}],
                    "recommended": "normalize",
                })

        # 5b) variasi mirip (typo/singkatan) lewat fuzzy matching
        if HAS_ML and _is_stringy(s) and 2 <= nun <= 300 and infer_kind(s) == "categorical":
            fm = _fuzzy.variant_mapping(s)
            if fm:
                raw = s.astype("string")
                aff = raw.isin(list(fm.keys()))
                issues.append({
                    "id": f"fuzzy:{col}", "type": "fuzzy", "column": col, "severity": "low",
                    "title": f"Kolom '{col}': {len(fm)} variasi mirip (kemungkinan typo, {int(aff.sum())} nilai)",
                    "detail": "Pasangan: " + ", ".join(f"'{k}'→'{v}'" for k, v in list(fm.items())[:4]) + ". Periksa dulu: tidak semua nilai mirip itu sama.",
                    "count": int(aff.sum()), "pct": round(100 * aff.mean(), 1), "rows": _rows(df.index[aff]), "samples": _sample(df, df.index[aff], col),
                    "stats": {"mapping": fm},
                    "actions": [{"key": "normalize", "label": "Gabungkan ke variasi terbanyak"}, {"key": "keep", "label": "Biarkan (memang beda)"}],
                    "recommended": "keep",
                })

        # 6) kolom konstan
        if n > 1 and nun == 1 and not na.any():
            issues.append({
                "id": f"constant:{col}", "type": "constant", "column": col, "severity": "low",
                "title": f"Kolom '{col}' berisi satu nilai saja",
                "detail": "Tidak membawa informasi untuk analisis.",
                "count": n, "pct": 100.0, "rows": [], "samples": [], "stats": {},
                "actions": [{"key": "drop_column", "label": "Hapus kolom"}, {"key": "keep", "label": "Biarkan"}],
                "recommended": "drop_column",
            })

    # 7) anomali multivariat (baris janggal bila dilihat gabungan kolom) yang belum tertangkap outlier univariat
    if HAS_ML and 50 <= n <= 30000:
        try:
            ncols = _outliers.numeric_cols(df)
            if len(ncols) >= 2:
                _, _, _, flagged = _outliers.score_rows(df, ncols, "iforest", 0.02)
                uni = set()
                for i in issues:
                    if i["type"] == "outliers":
                        c = i["column"]
                        lo, hi = i["stats"]["lower"], i["stats"]["upper"]
                        uni |= set(df.index[(df[c] < lo) | (df[c] > hi)])
                ids = [i for i, f in zip(df.index, flagged) if f and i not in uni]
                if len(ids) >= 3:
                    issues.append({
                        "id": "anomaly", "type": "anomaly", "column": None, "severity": "low",
                        "title": f"{len(ids)} baris janggal secara multivariat",
                        "detail": f"Tidak ekstrem di satu kolom mana pun, tetapi kombinasi {len(ncols)} kolom numeriknya jarang (Isolation Forest). Lihat penjelasan per baris di tab Analitik ML.",
                        "count": len(ids), "pct": round(100 * len(ids) / n, 1), "rows": _rows(ids), "rows_all": [jsonable(i) for i in ids], "samples": [], "stats": {},
                        "actions": [{"key": "drop_rows", "label": f"Hapus {len(ids)} baris"}, {"key": "keep", "label": "Biarkan"}],
                        "recommended": "keep",
                    })
        except Exception:
            pass

    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (order[i["severity"]], i["id"]))
    for i in issues:
        i["ignored"] = i["id"] in ignored
    return issues


# ------------------------------------------------------------------ aksi
def _find(ds: Dataset, issue_id: str) -> dict:
    for i in detect_issues(ds.df):
        if i["id"] == issue_id:
            return i
    raise KeyError("Masalah ini sudah tidak ada (mungkin sudah dibersihkan).")


def apply_action(ds: Dataset, issue_id: str, action: str, value=None) -> str:
    issue = _find(ds, issue_id)
    if action not in {a["key"] for a in issue["actions"]}:
        raise ValueError(f"Aksi '{action}' tidak tersedia untuk masalah ini.")
    df, col, typ = ds.df, issue["column"], issue["type"]

    if action == "keep":
        ds.snapshot(f"Biarkan: {issue['title']}")
        ds.ignored.add(issue_id)
        return "Dibiarkan. Masalah ini disembunyikan dari daftar."

    ds.snapshot(f"{action}: {issue['title']}")
    new = df.copy()

    if action == "drop_column":
        new = new.drop(columns=[col])
        msg = f"Kolom '{col}' dihapus."
    elif action == "drop_rows":
        if typ == "duplicates":
            before = len(new)
            new = new.drop_duplicates(keep="first")
            msg = f"{before - len(new)} baris duplikat dihapus."
        else:
            if typ == "missing":
                mask = new[col].isna()
            elif typ == "outliers":
                lo, hi = issue["stats"]["lower"], issue["stats"]["upper"]
                mask = (new[col] < lo) | (new[col] > hi)
            elif typ == "mixed_type":
                mask = new[col].notna() & _to_number(new[col]).isna()
            elif typ == "anomaly":
                mask = new.index.isin(issue["rows_all"])
            else:
                raise ValueError("Aksi tidak berlaku.")
            msg = f"{int(mask.sum())} baris dihapus."
            new = new[~mask]
    elif action in ("fill_knn", "fill_iterative"):
        mask = new[col].isna()
        filled = _impute.impute_column(new, col, "knn" if action == "fill_knn" else "iterative")
        new[col] = new[col].astype("float64")
        new.loc[mask, col] = filled[mask]
        msg = f"{int(mask.sum())} nilai kosong diisi dengan imputasi {'KNN' if action == 'fill_knn' else 'iteratif (MICE)'}."
    elif action in ("mask", "hash"):
        kind = issue["stats"].get("jenis", "")
        fn = (lambda v: _pii.mask_value(v, kind)) if action == "mask" else _pii.pseudonym
        new[col] = new[col].map(lambda v: v if pd.isna(v) else fn(v)).astype("object")
        msg = f"Kolom '{col}' {'disamarkan' if action == 'mask' else 'diganti pseudonim (hash)'}."
    elif action in ("fill_mean", "fill_median", "fill_mode", "fill_value"):
        if typ == "outliers":
            lo, hi = issue["stats"]["lower"], issue["stats"]["upper"]
            mask = (new[col] < lo) | (new[col] > hi)
            if action == "fill_median":
                fill = new.loc[~mask, col].median()
            elif action == "fill_value":
                fill = _parse_value(new[col], value)
            else:
                raise ValueError("Aksi tidak berlaku.")
            new.loc[mask, col] = fill
            msg = f"{int(mask.sum())} outlier diganti dengan {jsonable(fill)}."
        else:
            mask = new[col].isna()
            if action == "fill_mean":
                fill = new[col].mean()
            elif action == "fill_median":
                fill = new[col].median()
            elif action == "fill_mode":
                fill = new[col].mode(dropna=True).iloc[0]
            else:
                fill = _parse_value(new[col], value)
            new[col] = _assign(new[col], mask, fill)
            msg = f"{int(mask.sum())} nilai kosong diisi dengan {jsonable(fill)}."
    elif action == "clip":
        lo, hi = issue["stats"]["lower"], issue["stats"]["upper"]
        new[col] = new[col].clip(lo, hi)
        msg = f"Nilai '{col}' dibatasi ke [{lo:.4g}, {hi:.4g}]."
    elif action == "convert_numeric":
        new[col] = _to_number(new[col]).astype("float64")
        if new[col].dropna().mod(1).eq(0).all():
            new[col] = new[col].astype("Int64")
        msg = f"Kolom '{col}' diubah menjadi angka."
    elif action == "normalize":
        mapping = issue["stats"]["mapping"]
        new[col] = new[col].replace(mapping)
        msg = f"{len(mapping)} variasi penulisan diseragamkan."
    else:
        raise ValueError("Aksi tidak dikenal.")

    ds.df = new
    return msg


def auto_clean(ds: Dataset) -> list[str]:
    """Terapkan rekomendasi default untuk semua masalah yang belum diabaikan (bisa di-undo per langkah)."""
    msgs = []
    for _ in range(50):  # aman dari loop tak hingga
        todo = [i for i in detect_issues(ds.df, ds.ignored) if not i["ignored"] and i["recommended"] != "keep"]
        if not todo:
            break
        i = todo[0]
        try:
            msgs.append(f"{i['title']} → {apply_action(ds, i['id'], i['recommended'])}")
        except (ValueError, KeyError) as e:
            ds.ignored.add(i["id"])
            msgs.append(f"{i['title']} → dilewati ({e})")
    return msgs


def _parse_value(s: pd.Series, value):
    if value is None or str(value).strip() == "":
        raise ValueError("Isi nilai pengganti terlebih dahulu.")
    if is_numeric(s):
        try:
            v = float(str(value).replace(",", "."))
        except ValueError:
            raise ValueError(f"'{value}' bukan angka, sedangkan kolom ini numerik.") from None
        return v
    return str(value)


def _assign(s: pd.Series, mask: pd.Series, fill) -> pd.Series:
    s = s.copy()
    if is_numeric(s) and pd.api.types.is_integer_dtype(s) and not float(fill).is_integer():
        s = s.astype("float64")
    elif is_numeric(s) and pd.api.types.is_integer_dtype(s):
        fill = int(fill)
    s[mask] = fill
    return s


# ------------------------------------------------------------- edit manual
def edit_cell(ds: Dataset, row, col: str, value) -> None:
    if col not in ds.df.columns:
        raise KeyError(f"Kolom '{col}' tidak ada.")
    try:
        row = int(row) if not isinstance(row, (int, np.integer)) else row
    except (TypeError, ValueError):
        pass
    if row not in ds.df.index:
        raise KeyError(f"Baris {row} tidak ada.")
    s = ds.df[col]
    empty = value is None or (isinstance(value, str) and value.strip() == "")
    if empty:
        newv = np.nan if is_numeric(s) else pd.NA
    elif is_numeric(s):
        try:
            newv = float(str(value).replace(",", "."))
        except ValueError:
            raise ValueError(f"'{value}' bukan angka; kolom '{col}' bertipe numerik.") from None
    elif pd.api.types.is_bool_dtype(s):
        newv = str(value).strip().lower() in ("1", "true", "ya", "yes")
    else:
        newv = str(value)
    ds.snapshot(f"Edit manual: baris {row}, kolom '{col}'")
    new = ds.df.copy()
    if is_numeric(s) and not empty:
        if pd.api.types.is_integer_dtype(s) and not float(newv).is_integer():
            new[col] = new[col].astype("float64")
        elif pd.api.types.is_integer_dtype(s):
            newv = int(newv)
    new.at[row, col] = newv
    ds.df = new


def delete_rows(ds: Dataset, rows: list) -> int:
    idx = [r for r in rows if r in ds.df.index]
    if not idx:
        raise ValueError("Tidak ada baris yang cocok.")
    ds.snapshot(f"Hapus {len(idx)} baris terpilih")
    ds.df = ds.df.drop(index=idx)
    return len(idx)
