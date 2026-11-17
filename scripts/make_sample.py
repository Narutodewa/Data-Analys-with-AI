"""Membuat data/sample_sales.csv: data penjualan sintetis dengan masalah kualitas yang sengaja disisipkan."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300
dates = pd.date_range("2025-01-01", periods=n, freq="D")
region = rng.choice(["Jakarta", "Bandung", "Surabaya", "Medan"], n, p=[0.4, 0.25, 0.2, 0.15])
category = rng.choice(["Elektronik", "Fashion", "Makanan"], n)
price = rng.normal(150, 40, n).clip(20) * np.where(category == "Elektronik", 2.2, 1)
qty = rng.integers(1, 12, n)
ad = rng.normal(50, 15, n).clip(5)
revenue = price * qty * (1 + ad / 200) + rng.normal(0, 30, n)
df = pd.DataFrame({"tanggal": dates.strftime("%Y-%m-%d"), "wilayah": region, "kategori": category,
                   "harga": price.round(1), "jumlah": qty, "biaya_iklan": ad.round(1), "pendapatan": revenue.round(1),
                   "kepuasan": rng.integers(1, 6, n).astype(float)})
# --- masalah yang disisipkan ---
df.loc[rng.choice(n, 25, replace=False), "kepuasan"] = np.nan            # nilai kosong
df.loc[rng.choice(n, 8, replace=False), "biaya_iklan"] = np.nan
df.loc[[10, 77, 150], "pendapatan"] = [99999, 123456, -5000]           # outlier
df.loc[[5, 40], "wilayah"] = ["jakarta ", "BANDUNG"]                    # penulisan tidak konsisten
df["harga"] = df["harga"].astype(object)
df.loc[[3, 60], "harga"] = ["$250", "1,200"]                            # angka sebagai teks
df.loc[20, "harga"] = "tidak ada"                                       # bukan angka
df = pd.concat([df, df.iloc[[0, 1, 2, 3]]], ignore_index=True)          # duplikat
df["negara"] = "Indonesia"                                              # kolom konstan
df.to_csv("data/sample_sales.csv", index=False)
print(df.shape)
