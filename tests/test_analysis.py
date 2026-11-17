import json

import pandas as pd
import pytest

from app import ai, eda, recommend, tables
from app.charts import auto_charts, build_chart


def test_eda_overview_and_profiles(sample_ds):
    r = eda.run_eda(sample_ds.df)
    assert r["overview"]["rows"] == 304 and r["overview"]["duplicate_rows"] == 4
    prof = {c["name"]: c for c in r["columns"]}
    assert prof["tanggal"]["kind"] == "datetime"
    assert prof["wilayah"]["kind"] == "categorical"
    assert prof["pendapatan"]["kind"] == "numeric" and "hist" in prof["pendapatan"]
    json.dumps(r)  # harus bisa diserialisasi (tanpa NaN)


@pytest.mark.parametrize("spec", [
    {"type": "histogram", "x": "pendapatan"},
    {"type": "histogram", "x": "wilayah"},
    {"type": "bar", "x": "wilayah", "y": "pendapatan", "agg": "mean"},
    {"type": "bar", "x": "wilayah", "agg": "count"},
    {"type": "line", "x": "tanggal", "y": "pendapatan", "agg": "sum"},
    {"type": "scatter", "x": "biaya_iklan", "y": "pendapatan", "color": "kategori"},
    {"type": "box", "y": "pendapatan", "x": "kategori"},
    {"type": "pie", "x": "kategori", "y": "pendapatan"},
    {"type": "heatmap"},
])
def test_all_chart_types_build_and_serialize(sample_ds, spec):
    c = build_chart(sample_ds.df, spec)
    assert c["figure"]["data"]
    json.dumps(c, allow_nan=False)


def test_chart_validation_errors(sample_ds):
    with pytest.raises(ValueError):
        build_chart(sample_ds.df, {"type": "bar", "x": "nope"})
    with pytest.raises(ValueError):
        build_chart(sample_ds.df, {"type": "pizza"})
    with pytest.raises(ValueError):
        build_chart(sample_ds.df, {"type": "bar", "x": "wilayah", "y": "kategori", "agg": "sum"})


def test_auto_charts(sample_ds):
    assert len(auto_charts(sample_ds.df)) >= 4


def test_recommend_find_problems_without_goal(sample_ds):
    r = recommend.suggest(sample_ds.df)
    assert r["insights"] and r["charts"]
    titles = " ".join(i["title"] for i in r["insights"])
    assert "pendapatan" in titles
    json.dumps(r, allow_nan=False)


def test_recommend_with_goal_prioritizes_columns(sample_ds):
    r = recommend.suggest(sample_ds.df, "bagaimana tren pendapatan dari waktu ke waktu?")
    assert r["charts"][0]["spec"]["type"] == "line"
    r = recommend.suggest(sample_ds.df, "bandingkan pendapatan per wilayah")
    assert r["charts"][0]["spec"]["x"] == "wilayah"


def test_pivot_and_query(sample_ds):
    p = tables.pivot(sample_ds.df, ["wilayah"], "kategori", "jumlah", "sum")
    assert "wilayah" in p["columns"] and p["total"] >= 4
    q = tables.run_query(sample_ds.df, {"filters": [{"col": "jumlah", "op": ">=", "value": 10}], "groupby": ["wilayah"], "agg": {"pendapatan": "sum"}, "sort": {"by": "pendapatan", "desc": True}, "limit": 3})
    assert len(q) <= 3 and "pendapatan" in q.columns
    with pytest.raises(ValueError):
        tables.run_query(sample_ds.df, {"filters": [{"col": "x", "op": "==", "value": 1}]})


def test_formula_column_and_blocklist(sample_ds):
    tables.add_formula_column(sample_ds, "total", "jumlah * biaya_iklan")
    assert "total" in sample_ds.df.columns
    for bad in ["__import__('os').system('id')", "jumlah.__class__", "@x"]:
        with pytest.raises(ValueError):
            tables.add_formula_column(sample_ds, "x", bad)


def test_ai_offline_fallback(sample_ds, monkeypatch):
    monkeypatch.setattr(ai.config, "api_key", "")
    monkeypatch.setattr(ai.config, "provider", "anthropic")
    r = ai.ask(sample_ds, "apa yang menarik dari data ini?")
    assert r["mode"] == "offline" and r["answer"]


def test_ai_online_validates_and_executes_plan(sample_ds, monkeypatch):
    monkeypatch.setattr(ai.config, "api_key", "k")
    plan = {"answer": "Jakarta terbesar.", "charts": [
        {"title": "ok", "type": "bar", "x": "wilayah", "y": "pendapatan", "agg": "sum"},
        {"title": "buruk", "type": "bar", "x": "kolom_palsu", "y": None, "agg": "sum"}],
        "query": {"groupby": ["wilayah"], "agg": {"pendapatan": "sum"}, "sort": {"by": "pendapatan", "desc": True}, "limit": 2}}
    seen = {}

    def fake_complete(system, messages, max_tokens=0, image_b64=None):
        seen["prompt"] = messages[-1]["content"]
        return "Berikut:\n" + json.dumps(plan)

    monkeypatch.setattr(ai, "complete", fake_complete)
    r = ai.ask(sample_ds, "wilayah mana terbesar?")
    assert r["mode"] == "ai" and len(r["charts"]) == 1  # diagram dengan kolom palsu dibuang
    assert len(r["table"]["rows"]) == 2
    assert "sample_rows" in seen["prompt"]
    monkeypatch.setattr(ai.config, "send_sample", False)
    ai.ask(sample_ds, "x")
    assert "sample_rows" not in seen["prompt"]  # privasi: sampel tidak dikirim bila dimatikan
