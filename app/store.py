"""Penyimpanan dataset di memori (satu proses, satu pengguna) dengan riwayat undo."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import pandas as pd

MAX_HISTORY = 30
MAX_DATASETS = 50


@dataclass
class Dataset:
    name: str
    df: pd.DataFrame
    kind: str = "table"  # table | nosql | text | image
    meta: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    ignored: set = field(default_factory=set)
    history: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.original = self.df.copy()

    # --- riwayat -----------------------------------------------------
    def snapshot(self, label: str) -> None:
        self.history.append((self.df.copy(), set(self.ignored), label))
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        self.log.append(label)

    def undo(self) -> str:
        if not self.history:
            raise ValueError("Tidak ada langkah yang bisa dibatalkan.")
        self.df, self.ignored, label = self.history.pop()
        if self.log:
            self.log.pop()
        return label

    def reset(self) -> None:
        self.df = self.original.copy()
        self.ignored = set()
        self.history = []
        self.log = []

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "rows": int(len(self.df)),
            "cols": int(self.df.shape[1]),
            "columns": [str(c) for c in self.df.columns],
            "can_undo": bool(self.history),
            "log": list(self.log),
            "meta": {k: v for k, v in self.meta.items() if k in ("schema", "source", "note")},
        }


class Store:
    def __init__(self):
        self._items: dict[str, Dataset] = {}
        self._lock = threading.Lock()

    def add(self, ds: Dataset) -> Dataset:
        with self._lock:
            if len(self._items) >= MAX_DATASETS:
                raise ValueError(f"Maksimal {MAX_DATASETS} dataset sekaligus. Hapus sebagian dulu.")
            self._items[ds.id] = ds
        return ds

    def get(self, ds_id: str) -> Dataset:
        try:
            return self._items[ds_id]
        except KeyError:
            raise KeyError("Dataset tidak ditemukan.") from None

    def remove(self, ds_id: str) -> None:
        with self._lock:
            self._items.pop(ds_id, None)

    def list(self) -> list[Dataset]:
        return list(self._items.values())


store = Store()
