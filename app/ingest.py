"""Memasukkan data: CSV/TSV/TXT, Excel, JSON/JSONL (NoSQL), SQL/SQLite, Parquet, gambar, ZIP."""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import tempfile
import zipfile

import pandas as pd

from .images import IMAGE_EXT, images_to_dataset
from .store import Dataset

NA_TOKENS = ["", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "-", "--", "?"]
TEXT_EXT = {".csv", ".tsv", ".txt", ".dat"}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls"}
JSON_EXT = {".json", ".jsonl", ".ndjson"}
SQL_EXT = {".sql"}
SQLITE_EXT = {".db", ".sqlite", ".sqlite3"}
MAX_ZIP_FILES = 500


def _tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    cols = []
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"kolom_{i + 1}"
        if name == "_row":
            name = "row_"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df = df.copy()
    df.columns = cols
    return df


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


# ----------------------------------------------------------------- teks / csv
def _text_dataset(name: str, text: str) -> Dataset:
    lines = text.splitlines()
    df = pd.DataFrame({"line_no": range(1, len(lines) + 1), "text": lines})
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()
    return Dataset(name, df, kind="text", meta={"source": "text", "note": "Teks bebas dipecah per baris."})


def _sniff_sep(text: str) -> str | None:
    import csv
    try:
        return csv.Sniffer().sniff("\n".join(text.splitlines()[:50]), delimiters=",;\t|").delimiter
    except csv.Error:
        return None


def _load_delimited(name: str, data: bytes, force_table: bool) -> Dataset:
    text = _decode(data)
    sep = _sniff_sep(text)
    df = None
    if sep:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, na_values=NA_TOKENS, skipinitialspace=True)
        except Exception:
            df = None
    if df is None or df.shape[1] < 2:
        if force_table:
            raise ValueError("Tidak dapat mengenali pemisah kolom pada file ini (dukung: koma, titik koma, tab, |).")
        return _text_dataset(name, text)
    return Dataset(name, _tidy_columns(df), meta={"source": "csv"})


# ----------------------------------------------------------------------- excel
def _load_excel(name: str, data: bytes) -> list[Dataset]:
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, na_values=NA_TOKENS)
    out = []
    for sheet, df in sheets.items():
        if df.empty:
            continue
        label = name if len(sheets) == 1 else f"{name} [{sheet}]"
        out.append(Dataset(label, _tidy_columns(df), meta={"source": "excel"}))
    if not out:
        raise ValueError("Workbook kosong.")
    return out


# ------------------------------------------------------------------ json/nosql
def _stringify_nested(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (list, dict))).any():
            df[c] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    return df


def infer_doc_schema(docs: list, max_fields: int = 200) -> list[dict]:
    """Skema dokumen NoSQL: path field, tipe yang muncul, cakupan (%), contoh nilai."""
    stats: dict[str, dict] = {}

    def walk(obj, prefix):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        else:
            s = stats.setdefault(prefix, {"types": set(), "count": 0, "example": None})
            s["types"].add(type(obj).__name__ if obj is not None else "null")
            s["count"] += 1
            if s["example"] is None and obj not in (None, "", []):
                s["example"] = obj if not isinstance(obj, (list, dict)) else json.dumps(obj, ensure_ascii=False)[:80]

    for d in docs:
        if isinstance(d, dict):
            walk(d, "")
    n = max(len(docs), 1)
    rows = [
        {"field": k, "types": sorted(v["types"]), "coverage_pct": round(100 * v["count"] / n, 1), "example": v["example"]}
        for k, v in stats.items()
    ]
    rows.sort(key=lambda r: (-r["coverage_pct"], r["field"]))
    return rows[:max_fields]


def _load_json(name: str, data: bytes, ext: str) -> Dataset:
    text = _decode(data).strip()
    if not text:
        raise ValueError("File JSON kosong.")
    if ext in (".jsonl", ".ndjson") or (text.startswith("{") and "\n{" in text and not text.startswith("[")):
        try:
            docs = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON Lines tidak valid: {e}") from e
    else:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON tidak valid: {e}") from e
        docs = obj
        if isinstance(obj, dict):
            lists = [v for v in obj.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
            if len(lists) == 1:
                docs = lists[0]
            elif obj and all(isinstance(v, list) for v in obj.values()) and len({len(v) for v in obj.values()}) == 1:
                return Dataset(name, _tidy_columns(pd.DataFrame(obj)), meta={"source": "json"})
            else:
                docs = [obj]
    if not isinstance(docs, list) or not docs:
        raise ValueError("Struktur JSON tidak dikenali (harus berupa list dokumen/objek).")
    if not all(isinstance(d, dict) for d in docs):
        df = pd.DataFrame({"value": docs})
        return Dataset(name, df, meta={"source": "json"})
    schema = infer_doc_schema(docs)
    df = pd.json_normalize(docs, sep=".")
    df = _stringify_nested(df)
    nested = any(("." in r["field"]) or ("list" in r["types"]) for r in schema)
    return Dataset(
        name,
        _tidy_columns(df),
        kind="nosql" if nested or any(r["coverage_pct"] < 100 for r in schema) else "table",
        meta={"source": "json", "schema": schema, "note": "Dokumen di-flatten (field bersarang jadi kolom 'a.b'; array disimpan sebagai JSON)."},
    )


# ------------------------------------------------------------------------ sql
_SQL_DENY = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}


def _authorizer(action, *_):
    return sqlite3.SQLITE_DENY if action in _SQL_DENY else sqlite3.SQLITE_OK


def _sanitize_sql(sql: str) -> str:
    """Upaya terbaik: ubah dump MySQL umum agar bisa dibaca SQLite."""
    sql = re.sub(r"/\*!\d+.*?\*/;?", "", sql, flags=re.S)
    sql = re.sub(r"^\s*(LOCK|UNLOCK) TABLES.*?;\s*$", "", sql, flags=re.M | re.I)
    sql = re.sub(r"\)\s*ENGINE\s*=.*?;", ");", sql, flags=re.I | re.S)
    sql = re.sub(r"\bAUTO_INCREMENT\b", "", sql, flags=re.I)
    sql = re.sub(r"\bUNSIGNED\b", "", sql, flags=re.I)
    sql = sql.replace("`", '"')
    return sql


def _tables_from_conn(name: str, conn: sqlite3.Connection, source: str) -> list[Dataset]:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'")]
    out = []
    for t in tables:
        df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
        if len(df.columns):
            out.append(Dataset(f"{name} · {t}", _tidy_columns(df), meta={"source": source}))
    if not out:
        raise ValueError("Tidak ada tabel berisi data yang ditemukan di file SQL.")
    return out


def _load_sql(name: str, data: bytes) -> list[Dataset]:
    sql = _decode(data)
    conn = sqlite3.connect(":memory:")
    conn.set_authorizer(_authorizer)
    try:
        try:
            conn.executescript(sql)
        except sqlite3.Error:
            conn.close()
            conn = sqlite3.connect(":memory:")
            conn.set_authorizer(_authorizer)
            conn.executescript(_sanitize_sql(sql))
        return _tables_from_conn(name, conn, "sql")
    except sqlite3.Error as e:
        raise ValueError(f"SQL gagal dieksekusi di SQLite: {e}. (Dialek khusus MySQL/Postgres mungkin perlu diekspor ke CSV.)") from e
    finally:
        conn.close()


def _load_sqlite(name: str, data: bytes) -> list[Dataset]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "db.sqlite")
        with open(path, "wb") as f:
            f.write(data)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return _tables_from_conn(name, conn, "sqlite")
        except sqlite3.Error as e:
            raise ValueError(f"Bukan database SQLite yang valid: {e}") from e
        finally:
            conn.close()


# ------------------------------------------------------------------- dispatch
def load_file(name: str, data: bytes) -> list[Dataset]:
    ext = os.path.splitext(name.lower())[1]
    if ext in IMAGE_EXT:
        return [images_to_dataset(name, [(name, data)])]
    if ext in EXCEL_EXT:
        return _load_excel(name, data)
    if ext in JSON_EXT:
        return [_load_json(name, data, ext)]
    if ext in SQL_EXT:
        return _load_sql(name, data)
    if ext in SQLITE_EXT:
        return _load_sqlite(name, data)
    if ext == ".parquet":
        try:
            return [Dataset(name, _tidy_columns(pd.read_parquet(io.BytesIO(data))), meta={"source": "parquet"})]
        except ImportError as e:
            raise ValueError("Membaca .parquet memerlukan paket 'pyarrow'.") from e
    if ext == ".zip":
        return _load_zip(name, data)
    if ext in TEXT_EXT or ext == "":
        return [_load_delimited(name, data, force_table=ext in {".csv", ".tsv"})]
    raise ValueError(f"Format '{ext}' belum didukung.")


def _load_zip(name: str, data: bytes) -> list[Dataset]:
    out: list[Dataset] = []
    images: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members = [m for m in z.infolist() if not m.is_dir() and not os.path.basename(m.filename).startswith(".")]
        if len(members) > MAX_ZIP_FILES:
            raise ValueError(f"ZIP berisi lebih dari {MAX_ZIP_FILES} file.")
        if sum(m.file_size for m in members) > 500 * 1024 * 1024:
            raise ValueError("Isi ZIP terlalu besar (>500 MB setelah diekstrak).")
        for m in members:
            base = os.path.basename(m.filename)
            ext = os.path.splitext(base.lower())[1]
            blob = z.read(m)
            if ext in IMAGE_EXT:
                images.append((base, blob))
            else:
                try:
                    out.extend(load_file(f"{name}/{base}", blob))
                except ValueError:
                    continue
    if images:
        out.append(images_to_dataset(f"{name} (gambar)", images))
    if not out:
        raise ValueError("Tidak ada file yang dapat dibaca di dalam ZIP.")
    return out


def load_pasted(text: str, name: str = "tempelan") -> list[Dataset]:
    """Teks yang ditempel: otomatis deteksi JSON / SQL / CSV / teks bebas."""
    s = text.strip()
    if not s:
        raise ValueError("Teks kosong.")
    data = text.encode("utf-8")
    if s[0] in "[{":
        try:
            json.loads(s)
            return [_load_json(name, data, ".json")]
        except (json.JSONDecodeError, ValueError):
            if "\n" in s:
                try:
                    return [_load_json(name, data, ".jsonl")]
                except ValueError:
                    pass
    if re.match(r"(?is)^\s*(create|insert|drop|pragma|begin)\b", s):
        return _load_sql(name, data)
    return [_load_delimited(name, data, force_table=False)]
