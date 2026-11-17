import io
import json
import zipfile

import pandas as pd
import pytest

from app import ingest
from tests.conftest import make_png


def test_csv_semicolon_and_na_tokens():
    ds = ingest.load_file("a.csv", b"a;b\n1;x\n2;-\n3;N/A\n")[0]
    assert list(ds.df.columns) == ["a", "b"]
    assert ds.df["b"].isna().sum() == 2


def test_txt_free_text_becomes_text_dataset():
    ds = ingest.load_file("note.txt", b"halo dunia\nini baris dua\n")[0]
    assert ds.kind == "text" and "n_words" in ds.df.columns


def test_excel_multi_sheet(tmp_path):
    p = tmp_path / "x.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="S1", index=False)
        pd.DataFrame({"b": [3]}).to_excel(w, sheet_name="S2", index=False)
    out = ingest.load_file("x.xlsx", p.read_bytes())
    assert len(out) == 2 and "[S1]" in out[0].name


def test_json_nosql_flatten_and_schema():
    docs = [{"id": 1, "user": {"name": "a", "age": 3}, "tags": ["x", "y"]}, {"id": 2, "user": {"name": "b"}}]
    ds = ingest.load_file("d.json", json.dumps(docs).encode())[0]
    assert ds.kind == "nosql"
    assert "user.name" in ds.df.columns
    cov = {r["field"]: r["coverage_pct"] for r in ds.meta["schema"]}
    assert cov["user.age"] == 50.0
    assert ds.df["tags"].iloc[0] == '["x", "y"]'  # array jadi string -> bisa di-hash (duplikat dsb)


def test_jsonl():
    ds = ingest.load_file("d.jsonl", b'{"a":1}\n{"a":2}\n')[0]
    assert len(ds.df) == 2


def test_sql_dump_to_tables():
    sql = b"CREATE TABLE t(id INTEGER, n TEXT); INSERT INTO t VALUES (1,'a'),(2,'b');"
    out = ingest.load_file("d.sql", sql)
    assert len(out) == 1 and len(out[0].df) == 2


def test_sql_attach_is_denied(tmp_path):
    evil = f"ATTACH DATABASE '{tmp_path}/pwn.db' AS x; CREATE TABLE x.t(a);".encode()
    with pytest.raises(ValueError):
        ingest.load_file("e.sql", evil)
    assert not (tmp_path / "pwn.db").exists()


def test_images_to_feature_table():
    ds = ingest.load_file("a.png", make_png())[0]
    assert ds.kind == "image"
    assert {"width", "height", "brightness", "sharpness", "dominant_color"} <= set(ds.df.columns)
    assert ds.df.loc[0, "dominant_color"].startswith("#")
    assert "a.png" in ds.meta["thumbs"]


def test_zip_with_csv_and_images():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("t.csv", "a,b\n1,2\n3,4\n")
        z.writestr("p/1.png", make_png())
        z.writestr("p/2.png", make_png((0, 0, 255)))
    out = ingest.load_file("z.zip", buf.getvalue())
    assert {d.kind for d in out} == {"table", "image"}
    assert len([d for d in out if d.kind == "image"][0].df) == 2


def test_unknown_extension():
    with pytest.raises(ValueError):
        ingest.load_file("a.exe", b"x")


def test_paste_detects_json_sql_csv():
    assert ingest.load_pasted('[{"a":1},{"a":2}]')[0].df.shape == (2, 1)
    assert len(ingest.load_pasted("CREATE TABLE t(a); INSERT INTO t VALUES (1);")[0].df) == 1
    assert ingest.load_pasted("a,b\n1,2\n3,4")[0].df.shape == (2, 2)
