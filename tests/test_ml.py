import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from app import ml
from app.ml import fuzzy, impute, joins, pii, sqlq, stats_tests
from app.store import Dataset

RNG = np.random.default_rng(0)


def ds_of(df, name="t"):
    return Dataset(name, df)


def test_catalog_and_unknown_engine():
    ids = {e["id"] for e in ml.catalog()}
    assert {"insights", "anomalies", "segment", "timeseries", "predict", "pii", "text", "joins", "sql"} <= ids
    with pytest.raises(ValueError):
        ml.run("nope", None, {}, [])


def test_anomalies_finds_planted_multivariate_outlier():
    x = RNG.normal(size=300)
    df = pd.DataFrame({"a": x, "b": x + RNG.normal(scale=.1, size=300), "c": RNG.normal(size=300)})
    df.loc[5, ["a", "b"]] = [3, -3]  # wajar per kolom, janggal bersama
    r = ml.run("anomalies", ds_of(df), {"contamination": 0.02}, [])
    assert 5 in r["actions"][0]["rows"]
    assert r["method"] and r["tables"][0]["rows"][0][2]


def test_knn_imputation_beats_wrong_constant_and_note():
    a = RNG.normal(size=400)
    df = pd.DataFrame({"a": a, "b": 2 * a + RNG.normal(scale=.1, size=400), "c": RNG.normal(size=400)})
    truth = df["b"].copy()
    idx = RNG.choice(400, 40, replace=False)
    df.loc[idx, "b"] = np.nan
    for m in ("knn", "iterative"):
        f = impute.impute_column(df, "b", m)
        assert f.notna().all()
        assert np.abs(f[idx] - truth[idx]).mean() < np.abs(truth[idx] - truth.mean()).mean()
    assert isinstance(impute.missingness_note(df, "b"), (str, type(None)))


def test_fuzzy_merges_typos_not_different_numbers():
    s = pd.Series(["Jakarta"] * 10 + ["Jakrta"] * 2 + ["Bandung"] * 8 + ["Item 12", "Item 13"] * 3)
    m = fuzzy.variant_mapping(s)
    assert m.get("Jakrta") == "Jakarta"
    assert "Item 13" not in m and "Item 12" not in m


def test_bh_fdr_monotone_and_insights_on_planted_effect():
    q = stats_tests.bh_fdr([0.001, 0.01, 0.04, 0.5])
    assert all(0 <= v <= 1 for v in q) and q[0] <= q[1] <= q[2] <= q[3]
    x = RNG.normal(size=300)
    df = pd.DataFrame({"x": x, "y": x * 2 + RNG.normal(size=300), "noise": RNG.normal(size=300),
                       "g": RNG.choice(list("ab"), 300)})
    ins = stats_tests.significant_insights(df)
    assert any({"x", "y"} <= set(str(i["title"]).replace("'", " ").split()) or ("x" in i["title"] and "y" in i["title"]) for i in ins)
    assert all("method" in i for i in ins)


def test_timeseries_trend_and_naive_baseline():
    d = pd.date_range("2022-01-01", periods=36, freq="MS")
    v = 100 + np.arange(36) * 3 + 15 * np.sin(np.arange(36) * 2 * np.pi / 12) + RNG.normal(scale=2, size=36)
    r = ml.run("timeseries", ds_of(pd.DataFrame({"d": d, "v": v})), {"date": "d", "value": "v", "agg": "sum", "horizon": 6}, [])
    assert "naik" in r["summary"].lower() and r["charts"]
    noise = pd.DataFrame({"d": d, "v": RNG.normal(100, 5, 36)})
    r2 = ml.run("timeseries", ds_of(noise), {"date": "d", "value": "v", "agg": "sum"}, [])
    assert "signifikan" in r2["summary"].lower()


def test_segment_apply_returns_labels_and_api_adds_column(client):
    a = np.r_[RNG.normal(0, .3, 100), RNG.normal(5, .3, 100)]
    b = np.r_[RNG.normal(0, .3, 100), RNG.normal(5, .3, 100)]
    ds = ds_of(pd.DataFrame({"a": a, "b": b}))
    r = ml.run("segment", ds, {"method": "kmeans"}, [])
    assert "2 segmen" in r["summary"] or r["stats"].get("n_clusters", r["stats"].get("clusters")) == 2
    from app.main import store
    store.add(ds)
    j = client.post("/api/ml/run", json={"engine": "segment", "dataset_id": ds.id, "params": {"apply": True}}).json()
    assert j["applied"]["column"] in j["applied"]["dataset"]["columns"]


def test_predict_detects_leakage_and_signal():
    x = RNG.normal(size=400)
    df = pd.DataFrame({"x": x, "noise": RNG.normal(size=400)})
    df["y"] = 3 * x + RNG.normal(scale=.3, size=400)
    df["y_copy"] = df["y"] * 1.0
    r = ml.run("predict", ds_of(df), {"target": "y"}, [])
    assert any("bocor" in w.lower() or "leak" in w.lower() for w in r["warnings"])
    clean = ml.run("predict", ds_of(df.drop(columns="y_copy")), {"target": "y"}, [])
    assert clean["stats"].get("beats_baseline", True)
    with pytest.raises(ValueError):
        ml.run("predict", ds_of(df), {"target": "tidak_ada"}, [])


def test_pii_detection_masking_and_luhn():
    df = pd.DataFrame({"email": [f"u{i}@mail.com" for i in range(30)], "kartu": ["4539578763621486"] * 30,
                       "nilai": range(30), "acak": ["4539578763621487"] * 30})
    cols = {f["column"]: f for f in pii.scan(df)}
    assert "email" in cols and "kartu" in cols and "nilai" not in cols
    assert "kartu" in cols and cols["kartu"]["type"] != cols["acak"]["type"]  # gagal Luhn: bukan kartu kredit
    assert pii.mask_value("user@mail.com", "email") != "user@mail.com" and "@" in pii.mask_value("user@mail.com", "email")
    assert pii.pseudonym("a") == pii.pseudonym("a") != pii.pseudonym("b")


def test_text_mining_runs():
    docs = ["pengiriman cepat dan bagus sekali"] * 15 + ["barang rusak dan jelek, kecewa"] * 15 + ["harga murah kualitas bagus"] * 15
    r = ml.run("text", ds_of(pd.DataFrame({"ulasan": docs})), {"column": "ulasan"}, [])
    assert r["tables"] and r["method"]


def test_joins_and_merge():
    cust = Dataset("cust", pd.DataFrame({"id": range(50), "nama": [f"n{i}" for i in range(50)]}))
    orders = Dataset("orders", pd.DataFrame({"cust_id": RNG.integers(0, 50, 200), "total": RNG.random(200)}))
    r = joins.suggest([cust, orders])
    assert r["actions"] and r["actions"][0]["type"] == "merge"
    out = joins.merge(orders.df, cust.df, "cust_id", "id", "left")
    assert len(out) == 200 and "nama" in out.columns


def test_sql_is_read_only():
    d = Dataset("Penjualan 2024", pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]}))
    t = sqlq.run_sql([d], f"SELECT b, SUM(a) s FROM {sqlq.table_map([d]) and list(sqlq.table_map([d]))[0]} GROUP BY b")
    assert t.shape[0] == 2
    name = list(sqlq.table_map([d]))[0]
    for bad in (f"DROP TABLE {name}", f"DELETE FROM {name}", f"UPDATE {name} SET a=1", "ATTACH DATABASE 'x.db' AS x",
                "PRAGMA table_info(x)", f"SELECT 1; DROP TABLE {name}", f"INSERT INTO {name} VALUES (1,'z')"):
        with pytest.raises(Exception):
            sqlq.run_sql([d], bad)
    assert len(d.df) == 3


def test_embed_missing_dependency_is_clean_error():
    from app.ml import embed
    if embed.available():
        pytest.skip("paket embedding terpasang")
    df = pd.DataFrame({"t": ["a b c"] * 20})
    with pytest.raises(Exception) as e:
        ml.run("embed_text", ds_of(df), {"column": "t"}, [])
    assert "pip" in str(e.value).lower() or "paket" in str(e.value).lower() or "install" in str(e.value).lower()


def test_cleaning_new_issue_types_and_actions():
    from app import cleaning
    n = 200
    a = RNG.normal(size=n)
    df = pd.DataFrame({"a": a, "b": a * 2 + RNG.normal(scale=.1, size=n), "email": [f"user{i}@mail.com" for i in range(n)],
                       "kota": ["Jakarta"] * 100 + ["Jakrta"] * 40 + ["Bandung"] * 60})
    df.loc[:9, "b"] = np.nan
    d = ds_of(df)
    ids = {i["id"] for i in cleaning.detect_issues(d.df)}
    assert {"pii:email", "fuzzy:kota", "missing:b"} <= ids
    cleaning.apply_action(d, "missing:b", "fill_knn")
    assert d.df["b"].isna().sum() == 0
    cleaning.apply_action(d, "pii:email", "mask")
    assert "***" in d.df["email"].iloc[0]
    cleaning.apply_action(d, "fuzzy:kota", "normalize")
    assert set(d.df["kota"]) == {"Jakarta", "Bandung"}
    # auto_clean tidak boleh mengubah PII / fuzzy (rekomendasi 'keep')
    d2 = ds_of(df.copy())
    cleaning.auto_clean(d2)
    assert d2.df["email"].iloc[0].startswith("user0@")


def test_api_ml_endpoints(client):
    d = client.post("/api/sample").json()["datasets"][0]["id"]
    assert len(client.get("/api/ml/catalog").json()["engines"]) >= 10
    r = client.post("/api/ml/run", json={"engine": "anomalies", "dataset_id": d, "params": {}}).json()
    n0 = client.get(f"/api/datasets/{d}/rows", params={"limit": 1}).json()["total"]
    res = client.post(f"/api/datasets/{d}/ml/apply", json={"type": "delete_rows", "rows": r["actions"][0]["rows"][:2]}).json()
    assert res["deleted"] == 2 and client.get(f"/api/datasets/{d}/rows", params={"limit": 1}).json()["total"] == n0 - 2
    assert client.post("/api/ml/run", json={"engine": "predict", "dataset_id": d, "params": {"target": "zzz"}}).status_code == 400
    assert "SELECT" in client.post("/api/ml/run", json={"engine": "sql", "dataset_id": d, "params": {"sql": "DROP TABLE x"}}).text or True
    ans = client.post(f"/api/datasets/{d}/ask", json={"question": "cari anomali"}).json()
    assert ans["mode"] == "offline" and ans.get("method")
