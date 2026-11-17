"""SQL baca-saja di atas semua dataset (SQLite memori). Hanya SELECT; authorizer menolak semua selain membaca."""
from __future__ import annotations

import re
import sqlite3

import pandas as pd

from .common import need, result, table

ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
MAX_ROWS = 1000


def slug(name: str, used: set) -> str:
    base = re.sub(r"\W+", "_", re.sub(r"\.[A-Za-z0-9]+$", "", name)).strip("_").lower() or "tabel"
    if base[0].isdigit():
        base = "t_" + base
    out, i = base, 2
    while out in used:
        out, i = f"{base}_{i}", i + 1
    used.add(out)
    return out


def table_map(datasets) -> dict[str, object]:
    used: set = set()
    return {slug(d.name, used): d for d in datasets}


def schema_text(datasets) -> str:
    return "\n".join(f"{t}({', '.join(str(c) for c in d.df.columns)})" for t, d in table_map(datasets).items())


def _auth(action, *_):
    return sqlite3.SQLITE_OK if action in ALLOWED else sqlite3.SQLITE_DENY


def run_sql(datasets: list, sql: str, limit: int = MAX_ROWS) -> pd.DataFrame:
    s = sql.strip().rstrip(";")
    need(s, "SQL kosong.")
    need(";" not in s, "Hanya satu pernyataan SQL yang diizinkan.")
    need(re.match(r"(?is)^(select|with)\b", s) is not None, "Hanya perintah SELECT (atau WITH ... SELECT) yang diizinkan.")
    conn = sqlite3.connect(":memory:")
    try:
        for t, d in table_map(datasets).items():
            d.df.drop(columns=[c for c in d.df.columns if c == "_id"], errors="ignore").to_sql(t, conn, index=False)
        conn.set_authorizer(_auth)
        ops = [0]

        def guard():
            ops[0] += 1
            return 1 if ops[0] > 2000 else 0  # ~20 juta langkah VM

        conn.set_progress_handler(guard, 10000)
        try:
            return pd.read_sql_query(f"SELECT * FROM ({s}) LIMIT {int(limit)}", conn)
        except sqlite3.Error as e:
            raise ValueError(f"SQL gagal: {e}. Tabel yang tersedia:\n{schema_text(datasets)}") from e
    finally:
        conn.close()


def query(datasets, sql) -> dict:
    df = run_sql(datasets, sql)
    return result("Hasil SQL", "SQLite baca-saja", f"{len(df)} baris{' (dibatasi 1000)' if len(df) >= MAX_ROWS else ''}.",
                  assumptions=["Hanya SELECT. Semua dataset tersedia sebagai tabel; data asli tidak berubah.", "Tabel: " + ", ".join(table_map(datasets))],
                  tables=[table("Hasil", df, limit=MAX_ROWS)], stats={"rows": len(df), "columns": list(df.columns)})
