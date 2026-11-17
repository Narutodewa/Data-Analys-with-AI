"""FastAPI: REST API + file statis untuk antarmuka web."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, ai, cleaning, eda, ingest, ml, recommend, tables
from .ml import joins as ml_joins
from .charts import CHART_TYPES, AGGS, auto_charts, build_chart
from .store import Dataset, store
from .utils import jsonable

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

app = FastAPI(title="Data Analyst with AI", version=__version__)


@app.exception_handler(KeyError)
async def _keyerr(_: Request, e: KeyError):
    return JSONResponse({"detail": str(e.args[0]) if e.args else "Tidak ditemukan"}, status_code=404)


@app.exception_handler(ValueError)
async def _valerr(_: Request, e: ValueError):
    return JSONResponse({"detail": str(e)}, status_code=400)


@app.exception_handler(ai.AIError)
async def _aierr(_: Request, e: ai.AIError):
    return JSONResponse({"detail": str(e)}, status_code=502)


def _ds(ds_id: str) -> Dataset:
    return store.get(ds_id)


# ---------------------------------------------------------------- umum / AI
@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "chart_types": CHART_TYPES, "aggs": AGGS, "ai": ai.config.public()}


class AIConfigIn(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    send_sample: bool | None = None


@app.get("/api/ai/status")
def ai_status():
    return ai.config.public()


@app.post("/api/ai/config")
def ai_config(body: AIConfigIn):
    """Menyimpan pengaturan di memori proses saja (hilang saat server restart; key tidak pernah dikembalikan)."""
    c = ai.config
    if body.provider:
        if body.provider not in ("anthropic", "openai"):
            raise ValueError("provider harus 'anthropic' atau 'openai'.")
        if body.provider != c.provider:
            c.provider = body.provider
            c._load_key()
    if body.model is not None:
        c.model = body.model.strip()
    if body.base_url:
        c.base_url = body.base_url.strip()
    if body.api_key:
        c.api_key = body.api_key.strip()
    if body.send_sample is not None:
        c.send_sample = body.send_sample
    return c.public()


# ------------------------------------------------------------------ dataset
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    created, errors = [], []
    for f in files:
        data = await f.read()
        if len(data) > MAX_BYTES:
            errors.append({"file": f.filename, "error": f"Ukuran melebihi batas {MAX_BYTES // 1024 // 1024} MB."})
            continue
        try:
            for ds in ingest.load_file(f.filename or "data", data):
                store.add(ds)
                created.append(ds.summary())
        except Exception as e:  # tampilkan pesan ramah, jangan 500
            errors.append({"file": f.filename, "error": str(e)})
    # beberapa gambar yang diunggah bersamaan digabung menjadi satu dataset gambar
    imgs = [c for c in created if c["kind"] == "image"]
    if len(imgs) > 1:
        merged = _merge_images([store.get(c["id"]) for c in imgs])
        for c in imgs:
            store.remove(c["id"])
        store.add(merged)
        created = [c for c in created if c["kind"] != "image"] + [merged.summary()]
    return {"datasets": created, "errors": errors}


def _merge_images(items: list[Dataset]) -> Dataset:
    df = pd.concat([d.df for d in items], ignore_index=True)
    df = df.drop_duplicates(subset="filename")
    meta = {"source": "image", "note": items[0].meta.get("note"), "thumbs": {}, "images": {}}
    for d in items:
        meta["thumbs"].update(d.meta["thumbs"])
        meta["images"].update(d.meta["images"])
    return Dataset(f"Kumpulan gambar ({len(df)})", df, kind="image", meta=meta)


class PasteIn(BaseModel):
    text: str
    name: str | None = None


@app.post("/api/paste")
def paste(body: PasteIn):
    out = []
    for ds in ingest.load_pasted(body.text, body.name or "tempelan"):
        store.add(ds)
        out.append(ds.summary())
    return {"datasets": out, "errors": []}


@app.post("/api/sample")
def sample():
    p = ROOT / "data" / "sample_sales.csv"
    if not p.exists():
        raise KeyError("File contoh tidak ditemukan.")
    ds = ingest.load_file("contoh_penjualan.csv", p.read_bytes())[0]
    store.add(ds)
    return {"datasets": [ds.summary()], "errors": []}


@app.get("/api/datasets")
def list_datasets():
    return {"datasets": [d.summary() for d in store.list()]}


@app.get("/api/datasets/{ds_id}")
def get_dataset(ds_id: str):
    return _ds(ds_id).summary()


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: str):
    store.remove(ds_id)
    return {"ok": True}


@app.get("/api/datasets/{ds_id}/rows")
def rows(ds_id: str, offset: int = 0, limit: int = 100, sort: str | None = None, desc: bool = False, q: str | None = None, ids: str | None = None):
    df = _ds(ds_id).df
    if ids:
        try:
            want = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise ValueError("ids harus berupa angka dipisah koma.") from None
        df = df.loc[df.index.intersection(want)]
    return tables.page(df, offset, limit, sort, desc, q)


class CellIn(BaseModel):
    row: int | str
    col: str
    value: str | float | int | None = None


@app.put("/api/datasets/{ds_id}/cell")
def put_cell(ds_id: str, body: CellIn):
    ds = _ds(ds_id)
    cleaning.edit_cell(ds, body.row, body.col, body.value)
    return {"ok": True, "dataset": ds.summary()}


class RowsIn(BaseModel):
    rows: list[int]


@app.post("/api/datasets/{ds_id}/rows/delete")
def del_rows(ds_id: str, body: RowsIn):
    ds = _ds(ds_id)
    n = cleaning.delete_rows(ds, body.rows)
    return {"deleted": n, "dataset": ds.summary()}


class FormulaIn(BaseModel):
    name: str
    expr: str


@app.post("/api/datasets/{ds_id}/columns")
def add_column(ds_id: str, body: FormulaIn):
    ds = _ds(ds_id)
    tables.add_formula_column(ds, body.name, body.expr)
    return {"dataset": ds.summary()}


class DropColsIn(BaseModel):
    cols: list[str]


@app.post("/api/datasets/{ds_id}/columns/drop")
def drop_cols(ds_id: str, body: DropColsIn):
    ds = _ds(ds_id)
    missing = [c for c in body.cols if c not in ds.df.columns]
    if missing:
        raise KeyError(f"Kolom tidak ada: {', '.join(missing)}")
    ds.snapshot(f"Hapus kolom {', '.join(body.cols)}")
    ds.df = ds.df.drop(columns=body.cols)
    return {"dataset": ds.summary()}


@app.post("/api/datasets/{ds_id}/undo")
def undo(ds_id: str):
    ds = _ds(ds_id)
    label = ds.undo()
    return {"undone": label, "dataset": ds.summary()}


@app.post("/api/datasets/{ds_id}/reset")
def reset(ds_id: str):
    ds = _ds(ds_id)
    ds.reset()
    return {"dataset": ds.summary()}


# ----------------------------------------------------------------- cleaning
@app.get("/api/datasets/{ds_id}/issues")
def issues(ds_id: str, include_ignored: bool = False):
    ds = _ds(ds_id)
    all_ = cleaning.detect_issues(ds.df, ds.ignored)
    shown = all_ if include_ignored else [i for i in all_ if not i["ignored"]]
    return jsonable({"issues": shown, "ignored": sum(1 for i in all_ if i["ignored"]), "rows": len(ds.df)})


class CleanIn(BaseModel):
    issue_id: str
    action: str
    value: str | float | int | None = None


@app.post("/api/datasets/{ds_id}/clean")
def clean(ds_id: str, body: CleanIn):
    ds = _ds(ds_id)
    msg = cleaning.apply_action(ds, body.issue_id, body.action, body.value)
    return {"message": msg, "dataset": ds.summary()}


@app.post("/api/datasets/{ds_id}/clean/auto")
def clean_auto(ds_id: str):
    ds = _ds(ds_id)
    msgs = cleaning.auto_clean(ds)
    return {"messages": msgs, "dataset": ds.summary()}


@app.post("/api/datasets/{ds_id}/issues/unignore")
def unignore(ds_id: str):
    ds = _ds(ds_id)
    ds.ignored.clear()
    return {"ok": True}


@app.get("/api/datasets/{ds_id}/ai/clean-advice")
def clean_advice(ds_id: str):
    return ai.clean_advice(_ds(ds_id))


# --------------------------------------------------------------- eda / chart
@app.get("/api/datasets/{ds_id}/eda")
def get_eda(ds_id: str):
    return eda.run_eda(_ds(ds_id).df)


@app.get("/api/datasets/{ds_id}/auto-charts")
def get_auto_charts(ds_id: str):
    return {"charts": auto_charts(_ds(ds_id).df)}


class ChartIn(BaseModel):
    type: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    agg: str | None = "sum"
    bins: int | None = None
    title: str | None = None


@app.post("/api/datasets/{ds_id}/chart")
def make_chart(ds_id: str, body: ChartIn):
    return build_chart(_ds(ds_id).df, body.model_dump())


class RecommendIn(BaseModel):
    goal: str | None = None


@app.post("/api/datasets/{ds_id}/recommend")
def get_recommend(ds_id: str, body: RecommendIn):
    return recommend.suggest(_ds(ds_id).df, body.goal)


class PivotIn(BaseModel):
    index: list[str]
    columns: str | None = None
    values: str | None = None
    agg: str = "sum"


@app.post("/api/datasets/{ds_id}/pivot")
def make_pivot(ds_id: str, body: PivotIn):
    return tables.pivot(_ds(ds_id).df, body.index, body.columns, body.values, body.agg)


class AskIn(BaseModel):
    question: str
    history: list[dict] | None = None


@app.post("/api/datasets/{ds_id}/ask")
def ask(ds_id: str, body: AskIn):
    if not body.question.strip():
        raise ValueError("Pertanyaan kosong.")
    return ai.ask(_ds(ds_id), body.question, body.history, store.list())


# ------------------------------------------------------------- analitik ML
@app.get("/api/ml/catalog")
def ml_catalog():
    return {"engines": ml.catalog()}


class MLRunIn(BaseModel):
    engine: str
    dataset_id: str | None = None
    params: dict | None = None


@app.post("/api/ml/run")
def ml_run(body: MLRunIn):
    """Satu pintu untuk semua mesin analitik. Mesin 'segment' dengan params.apply=true menambahkan kolom segmen ke data."""
    ds = store.get(body.dataset_id) if body.dataset_id else None
    params = dict(body.params or {})
    res = ml.run(body.engine, ds, params, store.list())
    if body.engine == "segment" and params.get("apply"):
        labels = res.pop("labels", {})
        name = "segmen"
        while name in ds.df.columns:
            name += "_2"
        ds.snapshot(f"Tambah kolom '{name}' (segmentasi)")
        new = ds.df.copy()
        new[name] = pd.Series({k: v for k, v in labels.items()}).rename(index=lambda k: int(k) if str(k).lstrip("-").isdigit() else k).reindex(new.index)
        ds.df = new
        res["applied"] = {"column": name, "dataset": ds.summary()}
    res.get("stats", {}).pop("labels", None)
    return res


class MLApplyIn(BaseModel):
    type: str
    rows: list[int] | None = None
    filenames: list[str] | None = None


@app.post("/api/datasets/{ds_id}/ml/apply")
def ml_apply(ds_id: str, body: MLApplyIn):
    ds = _ds(ds_id)
    if body.type == "delete_rows":
        n = cleaning.delete_rows(ds, body.rows or [])
    elif body.type == "delete_rows_by_filename" and "filename" in ds.df.columns:
        rows = ds.df.index[ds.df["filename"].isin(body.filenames or [])].tolist()
        n = cleaning.delete_rows(ds, rows)
    else:
        raise ValueError("Jenis aksi tidak dikenal.")
    return {"deleted": n, "dataset": ds.summary()}


class MergeIn(BaseModel):
    left_id: str
    right_id: str
    left_on: str
    right_on: str
    how: str = "left"
    name: str | None = None


@app.post("/api/merge")
def merge(body: MergeIn):
    l, r = _ds(body.left_id), _ds(body.right_id)
    out = ml_joins.merge(l.df, r.df, body.left_on, body.right_on, body.how)
    new = Dataset(body.name or f"{l.name} + {r.name}", out, meta={"source": "merge", "note": f"Gabungan {body.how} join pada {body.left_on} = {body.right_on}."})
    store.add(new)
    return {"datasets": [new.summary()], "errors": []}


# -------------------------------------------------------------------- gambar
@app.get("/api/datasets/{ds_id}/thumb")
def thumb(ds_id: str, name: str):
    ds = _ds(ds_id)
    t = (ds.meta.get("thumbs") or {}).get(name)
    if t is None:
        raise KeyError("Thumbnail tidak ada.")
    return Response(t, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/datasets/{ds_id}/ai/describe-image")
def describe_image(ds_id: str, name: str):
    return {"description": ai.describe_image(_ds(ds_id), name)}


# -------------------------------------------------------------------- ekspor
@app.get("/api/datasets/{ds_id}/export")
def export(ds_id: str, fmt: str = Query("csv", pattern="^(csv|xlsx|json)$")):
    ds = _ds(ds_id)
    base = Path(ds.name).stem.replace(" ", "_")[:40] or "data"
    if fmt == "csv":
        body, mt = ds.df.to_csv(index=False).encode("utf-8-sig"), "text/csv"
    elif fmt == "json":
        body, mt = ds.df.to_json(orient="records", force_ascii=False, indent=1).encode(), "application/json"
    else:
        buf = io.BytesIO()
        ds.df.to_excel(buf, index=False)
        body, mt = buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return Response(body, media_type=mt, headers={"Content-Disposition": f'attachment; filename="{base}_clean.{fmt}"'})


# ------------------------------------------------------------------ frontend
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
