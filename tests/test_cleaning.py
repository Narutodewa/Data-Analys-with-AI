import numpy as np
import pandas as pd
import pytest

from app import cleaning
from app.store import Dataset


def ids(ds):
    return {i["id"]: i for i in cleaning.detect_issues(ds.df)}


def test_detects_all_planted_issues(sample_ds):
    found = ids(sample_ds)
    assert "duplicates" in found and found["duplicates"]["count"] == 4
    assert found["missing:kepuasan"]["count"] == 25
    assert "outliers:pendapatan" in found
    assert "mixed_type:harga" in found
    assert "text:wilayah" in found
    assert "constant:negara" in found


def test_missing_shows_stats_and_actions(sample_ds):
    i = ids(sample_ds)["missing:kepuasan"]
    assert i["stats"]["mean"] is not None and i["stats"]["median"] is not None
    keys = {a["key"] for a in i["actions"]}
    assert {"drop_rows", "fill_mean", "fill_median", "fill_value", "keep"} <= keys


def test_fill_mean_and_undo(sample_ds):
    mean = sample_ds.df["kepuasan"].mean()
    cleaning.apply_action(sample_ds, "missing:kepuasan", "fill_mean")
    assert sample_ds.df["kepuasan"].isna().sum() == 0
    assert sample_ds.df["kepuasan"].mean() == pytest.approx(mean)
    sample_ds.undo()
    assert sample_ds.df["kepuasan"].isna().sum() == 25


def test_drop_rows_missing(sample_ds):
    n = len(sample_ds.df)
    cleaning.apply_action(sample_ds, "missing:kepuasan", "drop_rows")
    assert len(sample_ds.df) == n - 25


def test_fill_value_validates_numeric(sample_ds):
    with pytest.raises(ValueError):
        cleaning.apply_action(sample_ds, "missing:kepuasan", "fill_value", "abc")
    cleaning.apply_action(sample_ds, "missing:kepuasan", "fill_value", "3")
    assert (sample_ds.df["kepuasan"] == 3).sum() >= 25


def test_convert_numeric_cleans_symbols(sample_ds):
    cleaning.apply_action(sample_ds, "mixed_type:harga", "convert_numeric")
    s = sample_ds.df["harga"]
    assert pd.api.types.is_numeric_dtype(s)
    assert s.loc[3] == 250 and s.loc[60] == 1200 and pd.isna(s.loc[20])


def test_outlier_clip_and_remove(sample_ds):
    i = ids(sample_ds)["outliers:pendapatan"]
    hi = i["stats"]["upper"]
    cleaning.apply_action(sample_ds, "outliers:pendapatan", "clip")
    assert sample_ds.df["pendapatan"].max() <= hi + 1e-9
    assert "outliers:pendapatan" not in ids(sample_ds)


def test_normalize_text(sample_ds):
    cleaning.apply_action(sample_ds, "text:wilayah", "normalize")
    assert set(sample_ds.df["wilayah"].unique()) == {"Jakarta", "Bandung", "Surabaya", "Medan"}


def test_duplicates_and_keep_hides_issue(sample_ds):
    cleaning.apply_action(sample_ds, "duplicates", "drop_rows")
    assert not sample_ds.df.duplicated().any()
    cleaning.apply_action(sample_ds, "constant:negara", "keep")
    assert "constant:negara" in sample_ds.ignored
    shown = [i for i in cleaning.detect_issues(sample_ds.df, sample_ds.ignored) if not i["ignored"]]
    assert all(i["id"] != "constant:negara" for i in shown)


def test_invalid_action_rejected(sample_ds):
    with pytest.raises(ValueError):
        cleaning.apply_action(sample_ds, "duplicates", "fill_mean")


def test_auto_clean_removes_recommended_issues(sample_ds):
    cleaning.auto_clean(sample_ds)
    left = [i for i in cleaning.detect_issues(sample_ds.df, sample_ds.ignored) if not i["ignored"] and i["recommended"] != "keep"]
    assert left == []
    assert len(sample_ds.history) >= 5  # setiap langkah bisa di-undo


def test_edit_cell_manual(sample_ds):
    cleaning.edit_cell(sample_ds, 0, "jumlah", "42")
    assert sample_ds.df.at[0, "jumlah"] == 42
    cleaning.edit_cell(sample_ds, 0, "jumlah", "2.5")  # int -> float, tidak crash
    assert sample_ds.df.at[0, "jumlah"] == 2.5
    cleaning.edit_cell(sample_ds, 1, "jumlah", "")
    assert pd.isna(sample_ds.df.at[1, "jumlah"])
    with pytest.raises(ValueError):
        cleaning.edit_cell(sample_ds, 0, "jumlah", "abc")
    with pytest.raises(KeyError):
        cleaning.edit_cell(sample_ds, 99999, "jumlah", "1")


def test_clean_dataset_has_no_false_positives():
    df = pd.DataFrame({"a": np.arange(50), "b": ["x", "y"] * 25})
    assert cleaning.detect_issues(df) == []
