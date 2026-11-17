"""Penambangan teks ringan tanpa model besar: topik (TF-IDF + NMF), kata kunci, sentimen berbasis leksikon (kasar)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import layout, need, result, table

STOP = set("""yang dan di ke dari untuk dengan pada ini itu adalah atau juga saya kami kita tidak ada akan sudah belum bisa lebih sangat karena dalam oleh
the a an and or of to in is are was were be for on with this that it as at by from not have has had we you they i""".split())
POS = set("bagus baik puas senang suka mantap cepat ramah murah recommended rekomendasi hebat keren memuaskan good great love excellent nice happy fast best amazing perfect".split())
NEG = set("buruk jelek lambat mahal kecewa rusak lama parah mengecewakan payah kotor bad poor slow terrible awful broken worst disappointed late hate".split())


def _tokens(t: str):
    return [w for w in re.findall(r"[a-zA-ZÀ-ɏ]{3,}", str(t).lower()) if w not in STOP]


def analyze(df: pd.DataFrame, column: str, n_topics: int | None = None) -> dict:
    need(column in df.columns, f"Kolom '{column}' tidak ditemukan.")
    docs = df[column].dropna().astype(str)
    docs = docs[docs.str.len() >= 3]
    need(len(docs) >= 10, "Butuh minimal 10 dokumen teks.")
    vec = TfidfVectorizer(tokenizer=_tokens, lowercase=False, token_pattern=None, max_features=3000, min_df=2 if len(docs) > 30 else 1, max_df=0.9)
    X = vec.fit_transform(docs)
    need(X.shape[1] >= 3, "Kosakata terlalu sedikit untuk analisis topik.")
    k = int(n_topics or min(6, max(2, len(docs) // 25)))
    k = max(2, min(k, X.shape[1] - 1, 10))
    nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=400)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    dom = W.argmax(axis=1)
    rows = []
    for i in range(k):
        words = terms[np.argsort(-nmf.components_[i])[:8]]
        ex = docs.iloc[np.where(dom == i)[0][:1]].tolist()
        rows.append({"topik": f"T{i + 1}", "kata_kunci": ", ".join(words), "dokumen": int((dom == i).sum()), "contoh": (ex[0][:100] if ex else "")})
    tok = docs.map(_tokens)
    pos = tok.map(lambda t: sum(w in POS for w in t))
    neg = tok.map(lambda t: sum(w in NEG for w in t))
    sent = np.where(pos > neg, "positif", np.where(neg > pos, "negatif", "netral"))
    sc = pd.Series(sent).value_counts()
    freq = pd.Series([w for t in tok for w in t]).value_counts().head(15)[::-1]
    charts = [
        {"title": "Kata teratas", "figure": {"data": [{"type": "bar", "orientation": "h", "x": freq.tolist(), "y": freq.index.tolist(), "marker": {"color": "#4f7cff"}}], "layout": layout("Kata paling sering", "frekuensi", None, margin={"l": 100, "r": 20, "t": 44, "b": 50})}},
        {"title": "Sentimen", "figure": {"data": [{"type": "pie", "labels": sc.index.tolist(), "values": sc.tolist(), "hole": .4}], "layout": layout("Sentimen (leksikon)")}},
    ]
    return result(f"Analisis teks '{column}'", "TF-IDF + NMF (topik), leksikon (sentimen)",
                  f"{len(docs)} dokumen dikelompokkan ke {k} topik. Topik terbesar: {rows[int(np.argmax([r['dokumen'] for r in rows]))]['kata_kunci']}. Sentimen: " + ", ".join(f"{a} {b}" for a, b in sc.items()) + ".",
                  assumptions=["Sentimen memakai leksikon kecil (ID/EN) sehingga KASAR: abaikan negasi dan sarkasme. Untuk akurasi lebih baik, gunakan embedding (lihat engine 'embed') atau tanya AI.",
                               "Topik adalah pengelompokan kata statistik; beri nama sendiri."],
                  warnings=["Daftar stopword singkat; kata umum bisa muncul."] if len(docs) < 30 else [],
                  tables=[table("Topik", pd.DataFrame(rows))], charts=charts, stats={"documents": len(docs), "topics": k, "sentiment": sc.to_dict()})
