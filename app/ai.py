"""Lapisan AI. Dua penyedia: Anthropic dan API kompatibel-OpenAI (OpenAI, Ollama, LM Studio, vLLM, dll).

Prinsip keamanan & privasi:
- AI TIDAK mengeksekusi kode. AI hanya mengembalikan spesifikasi terstruktur (diagram / kueri) yang divalidasi lalu dijalankan oleh backend.
- Yang dikirim ke penyedia: skema, statistik ringkas, dan (opsional) 5 baris contoh. Bukan seluruh data.
- Tanpa API key, semua fitur tetap jalan lewat heuristik offline (recommend.py).
"""
from __future__ import annotations

import base64
import json
import os
import re

import httpx
import pandas as pd

from . import cleaning, recommend
from . import ml
from .ml import sqlq
from .charts import AGGS, CHART_TYPES, build_chart
from .eda import column_profile
from .store import Dataset
from .tables import run_query
from .utils import jsonable

TIMEOUT = 90.0


class AIConfig:
    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "anthropic").lower()
        self.model = os.getenv("AI_MODEL", "")
        self.api_key = ""
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.send_sample = os.getenv("AI_SEND_SAMPLE", "true").lower() != "false"
        self._load_key()

    def _load_key(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY", "")

    @property
    def enabled(self) -> bool:
        # Ollama/LM Studio lokal tidak butuh key
        local = self.provider == "openai" and ("localhost" in self.base_url or "127.0.0.1" in self.base_url)
        return bool(self.api_key) or local

    @property
    def effective_model(self) -> str:
        return self.model or ("claude-sonnet-5-5" if self.provider == "anthropic" else "gpt-4o-mini")

    def public(self) -> dict:
        return {"enabled": self.enabled, "provider": self.provider, "model": self.effective_model,
                "base_url": self.base_url if self.provider == "openai" else None, "send_sample": self.send_sample}


config = AIConfig()


class AIError(RuntimeError):
    pass


# ------------------------------------------------------------------ transport
def _post(url, headers, payload):
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise AIError(f"Tidak bisa menghubungi penyedia AI: {e}") from e
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {})
            msg = msg.get("message") if isinstance(msg, dict) else msg
        except Exception:
            msg = r.text[:200]
        raise AIError(f"Penyedia AI menolak permintaan ({r.status_code}): {msg}")
    return r.json()


def complete(system: str, messages: list[dict], max_tokens: int = 1500, image_b64: str | None = None) -> str:
    if not config.enabled:
        raise AIError("AI belum dikonfigurasi.")
    msgs = [dict(m) for m in messages]
    if config.provider == "anthropic":
        if image_b64:
            msgs[-1]["content"] = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": msgs[-1]["content"]},
            ]
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": config.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                     {"model": config.effective_model, "max_tokens": max_tokens, "system": system, "messages": msgs})
        return "".join(b.get("text", "") for b in data.get("content", []))
    if image_b64:
        msgs[-1]["content"] = [
            {"type": "text", "text": msgs[-1]["content"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    headers = {"content-type": "application/json"}
    if config.api_key:
        headers["authorization"] = f"Bearer {config.api_key}"
    data = _post(config.base_url.rstrip("/") + "/chat/completions", headers,
                 {"model": config.effective_model, "max_tokens": max_tokens, "messages": [{"role": "system", "content": system}, *msgs]})
    return data["choices"][0]["message"]["content"] or ""


# ------------------------------------------------------------------- konteks
def dataset_context(ds: Dataset) -> str:
    df = ds.df
    cols = []
    for c in df.columns[:60]:
        p = column_profile(df[c], c)
        item = {k: p[k] for k in ("name", "kind", "missing_pct", "unique") if k in p}
        if p["kind"] == "numeric":
            item.update({k: round(p[k], 4) for k in ("min", "median", "mean", "max") if k in p})
        elif "top" in p:
            item["top"] = [t["value"] for t in p["top"][:5]]
        elif p["kind"] == "datetime":
            item.update({"min": p.get("min"), "max": p.get("max")})
        cols.append(item)
    ctx = {"dataset": ds.name, "type": ds.kind, "rows": len(df), "columns": cols}
    if config.send_sample:
        ctx["sample_rows"] = jsonable(df.head(5).astype(object).where(df.head(5).notna(), None).to_dict("records"))
    return json.dumps(ctx, ensure_ascii=False)


SYSTEM_ASK = """Anda adalah analis data senior. Jawab dalam bahasa pengguna (default Bahasa Indonesia), ringkas dan jujur tentang ketidakpastian.
Anda hanya melihat skema/statistik ringkas data, bukan seluruh data - jangan mengarang angka yang tidak ada di konteks.
Balas HANYA dengan satu objek JSON (tanpa markdown) berformat:
{"answer": "penjelasan singkat", 
 "charts": [{"title": "...", "type": "<%s>", "x": "kolom|null", "y": "kolom|null", "color": "kolom|null", "agg": "<%s>"}],
 "query": null | {"filters":[{"col":"..","op":"==|!=|>|>=|<|<=|contains|in|isnull|notnull","value":..}], "groupby":["kolom"], "agg":{"kolom":"sum|mean|median|min|max|count|nunique|std"}, "sort":{"by":"kolom","desc":true}, "limit":20}}
Aturan: pakai hanya nama kolom yang ada di konteks. Maksimal 3 diagram. Gunakan "query" bila pertanyaan butuh angka pasti (top-N, total per grup); sistem akan menjalankannya pada data penuh dan menampilkan hasilnya.

MESIN ANALITIK (statistik/ML yang menghitung; kamu TIDAK boleh menghitung sendiri). Tambahkan kunci opsional:
 "analysis": {"engine": "insights|anomalies|segment|timeseries|predict|pii|text|images_dup|joins", "params": {...}}
 - insights: temuan terverifikasi statistik {} ; anomalies: {"method":"iforest|lof|mad","columns":[..]} ; segment: {"method":"kmeans|hdbscan","k":int|null,"columns":[..]}
 - timeseries: {"date":"kolom","value":"kolom","agg":"sum|mean|median|min|max|count","horizon":int|null} ; predict: {"target":"kolom","features":[..]|null} (faktor yang memengaruhi / prediksi)
 - pii: {} ; text: {"column":"kolom"} ; images_dup: {} ; joins: {}
 "sql": null | "SELECT ..." (baca-saja, lintas dataset; nama tabel ada di konteks "tables").
Pakai analysis untuk pertanyaan tentang prediksi, faktor berpengaruh, anomali, segmentasi, tren/ramalan, data pribadi, topik teks. Setelah mesin berjalan, sistem akan memintamu menjelaskan hasilnya.
""" % ("|".join(CHART_TYPES), "|".join(AGGS))


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"answer": text, "charts": [], "query": None}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"answer": text, "charts": [], "query": None}


ENGINE_HINTS = [
    ("anomalies", ("anomali", "janggal", "aneh", "pencilan", "outlier multivariat")),
    ("segment", ("segmen", "klaster", "cluster", "kelompokkan", "pengelompokan")),
    ("pii", ("pii", "data pribadi", "sensitif", "privasi", "pribadi")),
    ("timeseries", ("ramal", "forecast", "prediksi ke depan", "musiman", "tren")),
    ("predict", ("prediksi", "memprediksi", "faktor", "memengaruhi", "mempengaruhi", "berpengaruh", "penyebab")),
    ("insights", ("temuan", "insight", "hubungan", "korelasi", "menarik", "pola")),
    ("images_dup", ("duplikat gambar", "gambar mirip", "gambar kembar")),
]


def route_offline(ds: Dataset, question: str) -> tuple[str, dict] | None:
    """Tanpa LLM: petakan pertanyaan ke mesin analitik lewat kata kunci + nama kolom yang disebut."""
    q = question.lower()
    mentioned = [c for c in ds.df.columns if str(c).lower().replace("_", " ") in q.replace("_", " ")]
    from .utils import infer_kind
    kinds = {c: infer_kind(ds.df[c]) for c in ds.df.columns}
    dts = [c for c in mentioned if kinds[c] == "datetime"] or [c for c, k in kinds.items() if k == "datetime"]
    nums = [c for c in mentioned if kinds[c] == "numeric"]
    for engine, words in ENGINE_HINTS:
        if any(w in q for w in words):
            if engine == "timeseries":
                if not dts or not (nums or [c for c, k in kinds.items() if k == "numeric"]):
                    continue
                return engine, {"date": dts[0], "value": (nums or [c for c, k in kinds.items() if k == "numeric"])[0], "agg": "sum"}
            if engine == "predict":
                if not mentioned:
                    continue
                return engine, {"target": mentioned[0]}
            if engine == "images_dup" and ds.kind != "image":
                continue
            return engine, {}
    return None


NARRATE_SYS = ("Kamu analis data. Tulis jawaban Bahasa Indonesia ringkas (3-6 kalimat) untuk pertanyaan pengguna, HANYA berdasarkan hasil mesin statistik/ML di bawah. "
               "Jangan menambah atau mengubah angka. Sebutkan peringatan/batasan penting (mis. tidak mengalahkan baseline, data kecil, korelasi bukan sebab-akibat). Tanpa markdown berat.")


def _engine_payload(res: dict) -> dict:
    keep = {k: v for k, v in (res.get("stats") or {}).items() if not isinstance(v, (list, dict)) or len(str(v)) < 300}
    return {"judul": res["title"], "metode": res["method"], "ringkasan": res["summary"], "asumsi": res["assumptions"], "peringatan": res["warnings"], "statistik": keep,
            "tabel": [{"judul": t["title"], "kolom": t["columns"], "baris": t["rows"][:8]} for t in res["tables"][:2]]}


def _table_of(res: dict):
    return None


def ask(ds: Dataset, question: str, history: list[dict] | None = None, all_datasets: list | None = None) -> dict:
    """Tanya jawab. LLM merencanakan (diagram/kueri/mesin), backend menghitung, LLM menarasikan. Tanpa LLM: pemetaan kata kunci ke mesin."""
    df = ds.df
    all_datasets = all_datasets or [ds]
    if not config.enabled:
        route = route_offline(ds, question)
        if route:
            try:
                res = ml.run(route[0], ds, route[1], all_datasets)
                ans = f"Mode offline (tanpa LLM): saya menjalankan mesin '{res['title']}'.\n{res['summary']}" + ("\nPeringatan: " + "; ".join(res["warnings"][:3]) if res["warnings"] else "")
                return {"answer": ans, "charts": res["charts"][:3], "table": None, "tables": res["tables"][:2], "method": res["method"], "mode": "offline"}
            except ValueError as e:
                return {"answer": f"Mesin '{route[0]}' tidak bisa dijalankan: {e}", "charts": [], "table": None, "mode": "offline"}
        rec = recommend.suggest(df, question)
        lines = [f"- {i['title']}" for i in rec["insights"][:5]]
        ans = ("Mode offline (AI belum dikonfigurasi), jadi saya memakai analisis otomatis. "
               + ("Temuan yang relevan:\n" + "\n".join(lines) if lines else "Belum ada temuan menonjol.")
               + "\n\nAktifkan LLM lewat tombol ⚙ untuk jawaban yang lebih fleksibel.")
        return {"answer": ans, "charts": rec["charts"][:3], "table": None, "mode": "offline"}

    ctx = json.loads(dataset_context(ds))
    ctx["tables"] = sqlq.schema_text(all_datasets).splitlines()
    msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-6:] if m.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": f"KONTEKS DATA:\n{json.dumps(ctx, ensure_ascii=False)}\n\nPERTANYAAN: {question}"})
    raw = complete(SYSTEM_ASK, msgs, max_tokens=1800)
    plan = _extract_json(raw)
    charts, errors, tables_out, method = [], [], [], None
    for spec in (plan.get("charts") or [])[:3]:
        try:
            spec = {k: spec.get(k) for k in ("title", "type", "x", "y", "color", "agg")}
            charts.append(build_chart(df, spec))
        except Exception as e:
            errors.append(str(e))
    table = None
    if plan.get("query"):
        try:
            t = run_query(df, plan["query"], limit=100)
            table = {"columns": list(t.columns), "rows": jsonable(t.astype(object).where(t.notna(), None).values.tolist())}
        except Exception as e:
            errors.append(f"Kueri gagal: {e}")
    if plan.get("sql"):
        try:
            t = sqlq.run_sql(all_datasets, str(plan["sql"]), limit=100)
            table = {"columns": [str(c) for c in t.columns], "rows": jsonable(t.astype(object).where(t.notna(), None).values.tolist())}
            method = "SQL baca-saja"
        except Exception as e:
            errors.append(f"SQL gagal: {e}")
    ans = str(plan.get("answer") or "")
    an = plan.get("analysis")
    if isinstance(an, dict) and an.get("engine") in ml.CATALOG:
        try:
            res = ml.run(an["engine"], ds, an.get("params") or {}, all_datasets)
            charts += res["charts"][:3]
            tables_out = res["tables"][:2]
            method = res["method"]
            try:  # LLM hanya menarasikan; angka dari mesin
                ans = complete(NARRATE_SYS, [{"role": "user", "content": f"PERTANYAAN: {question}\n\nHASIL MESIN:\n{json.dumps(_engine_payload(res), ensure_ascii=False)}"}], max_tokens=700).strip() or res["summary"]
            except AIError:
                ans = res["summary"]
            if res["warnings"]:
                ans += "\n\nPeringatan: " + "; ".join(res["warnings"][:3])
        except (ValueError, KeyError) as e:
            errors.append(f"Mesin '{an.get('engine')}' gagal: {e}")
    if errors and not charts and not table and not tables_out:
        ans += "\n\n(Catatan: saya tidak berhasil menjalankan usulan AI: " + "; ".join(errors[:2]) + ")"
    return {"answer": ans, "charts": charts, "table": table, "tables": tables_out, "method": method, "mode": "ai"}


def clean_advice(ds: Dataset) -> dict:
    issues = [i for i in cleaning.detect_issues(ds.df, ds.ignored) if not i["ignored"]]
    if not issues:
        return {"advice": "Tidak ada masalah kualitas data yang terdeteksi. 🎉", "mode": "offline"}
    brief = [{"id": i["id"], "title": i["title"], "detail": i["detail"], "recommended": i["recommended"], "stats": {k: v for k, v in i["stats"].items() if k != "mapping"}} for i in issues[:15]]
    if not config.enabled:
        return {"advice": "\n".join(f"• {i['title']} → disarankan: {i['recommended']}" for i in issues[:15]), "mode": "offline"}
    sys_ = "Anda analis data. Beri saran pembersihan data berprioritas, singkat, dalam Bahasa Indonesia. Jelaskan risiko tiap pilihan (misal isi rata-rata vs median vs hapus) dan akhiri dengan urutan langkah yang disarankan."
    txt = complete(sys_, [{"role": "user", "content": f"Dataset: {ds.name} ({len(ds.df)} baris)\nMasalah terdeteksi:\n{json.dumps(brief, ensure_ascii=False)}"}], max_tokens=900)
    return {"advice": txt, "mode": "ai"}


def describe_image(ds: Dataset, filename: str) -> str:
    img = (ds.meta.get("images") or {}).get(filename)
    if img is None:
        raise KeyError("Gambar tidak ditemukan.")
    if not config.enabled:
        raise AIError("Deskripsi gambar membutuhkan AI (aktifkan lewat tombol ⚙).")
    return complete("Anda analis citra. Deskripsikan gambar secara objektif dalam Bahasa Indonesia: objek utama, kondisi kualitas (buram, gelap, noise), dan hal yang perlu diperiksa.",
                    [{"role": "user", "content": "Deskripsikan gambar ini."}], max_tokens=500, image_b64=base64.b64encode(img).decode())
