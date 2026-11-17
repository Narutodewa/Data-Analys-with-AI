# 📊 Data Analyst with AI

Aplikasi web untuk analisis data yang bisa dipakai **dengan atau tanpa AI**: masukkan data (terstruktur maupun tidak), bersihkan dengan kontrol penuh, jelajahi, lalu biarkan aplikasi (dan AI) mencari masalah serta membuat visual otomatis.

> *English summary:* a self-hosted data-analysis web app (FastAPI + vanilla JS). Ingest CSV/Excel/JSON/SQL/images, interactively clean data (delete / fill mean / edit manually), run EDA, get auto-recommended charts, and chat with your data using Claude or any OpenAI-compatible model. Works fully offline without an API key.

## Fitur

| Fitur | Isi |
|---|---|
| **1. Memasukkan data** | CSV/TSV/TXT, Excel (semua sheet), JSON & JSON Lines (**NoSQL**: field bersarang di-flatten + skema & cakupan field), SQL dump & SQLite, Parquet, teks bebas, **gambar** (fitur visual: ukuran, kecerahan, ketajaman, warna dominan + thumbnail), ZIP campuran, atau tempel langsung. |
| **2. Data cleaning interaktif** | Deteksi otomatis: nilai kosong, duplikat, outlier (IQR), angka tersimpan sebagai teks (`$250`, `1,200`), penulisan tidak konsisten, kolom konstan. Tiap masalah punya tombol: **hapus**, **isi rata-rata / median / modus / nilai tertentu**, **batasi (winsorize)**, **edit manual** per sel, atau **biarkan**. Ada tombol "terapkan semua rekomendasi" dan **undo** tiap langkah. |
| **3. EDA** | Ringkasan baris/kolom/kosong/duplikat, profil tiap kolom (statistik, distribusi, top nilai), korelasi, dan diagram otomatis. |
| **4. Rekomendasi & visual otomatis** | *Punya masalah?* tulis pertanyaan → diagram yang relevan. *Belum tahu masalahnya?* kosongkan → aplikasi mencari korelasi kuat, tren waktu, perbedaan antar segmen, distribusi miring, outlier, data tidak seimbang. |
| **5. Tanya AI** | Tanya data dalam bahasa natural; AI membalas penjelasan + diagram + kueri (top-N, total per grup) yang dijalankan aplikasi pada data penuh. Deskripsi gambar dengan model vision. |
| **6. Ala Excel/Tableau** | Tabel dengan sort/cari/edit sel, kolom rumus (`harga * jumlah`), pivot table, pembuat grafik drag-select (histogram, bar, line, scatter, box, pie, heatmap), ekspor CSV/Excel/JSON. |

### Desain AI yang aman
- AI **tidak mengeksekusi kode**. Ia hanya mengembalikan spesifikasi terstruktur (diagram/kueri) yang divalidasi lalu dijalankan backend.
- Yang dikirim ke penyedia hanya **skema + statistik ringkas** (+ 5 baris contoh, bisa dimatikan). Bukan seluruh data.
- Tanpa API key semua fitur tetap jalan lewat heuristik (`app/recommend.py`).
- Penyedia: **Anthropic** atau **OpenAI-kompatibel** (OpenAI, Ollama, LM Studio, vLLM → data bisa tetap 100% lokal).

## Menjalankan

```bash
git clone https://github.com/Narutodewa/Data-Analys-with-AI.git
cd Data-Analys-with-AI
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # opsional: isi API key
uvicorn app.main:app --reload
```
Buka <http://localhost:8000>. Klik **Data contoh** untuk mencoba (berisi masalah yang sengaja disisipkan).

- Variabel `.env` tidak dibaca otomatis oleh uvicorn: gunakan `uvicorn app.main:app --env-file .env` atau ekspor manual. Key juga bisa diisi lewat tombol **⚙ Pengaturan AI** (disimpan di memori saja).
- Grafik memakai Plotly dari CDN. Untuk **offline**: `sh scripts/vendor_plotly.sh`.
- Docker: `docker build -t data-analyst . && docker run -p 8000:8000 --env-file .env data-analyst`

## Struktur

```
app/
  main.py        REST API + static files
  ingest.py      loader semua format (CSV, Excel, JSON/NoSQL, SQL, SQLite, ZIP, tempel)
  images.py      citra -> tabel fitur + thumbnail
  cleaning.py    deteksi masalah + aksi (hapus/isi/clip/normalisasi/edit manual)
  eda.py         profil kolom & korelasi
  charts.py      spesifikasi sederhana -> figure Plotly
  recommend.py   mesin temuan & rekomendasi heuristik (tanpa AI)
  tables.py      paging, pivot, kueri terstruktur, kolom rumus
  ai.py          penyedia AI + prompt + validasi hasil
  static/        antarmuka (HTML/CSS/JS, tanpa build step)
tests/           pytest (backend + API)
data/            data contoh;  scripts/ pembuat data contoh & vendor Plotly
docs/ROADMAP.md  rencana pengembangan
```

## Tes
```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Batasan saat ini (jujur)
- **Satu pengguna, data di memori**: cocok untuk penggunaan lokal. Jangan diekspos ke internet tanpa autentikasi (kolom rumus memakai `DataFrame.eval` dengan daftar blokir; SQL dijalankan di SQLite memori dengan `ATTACH` diblokir).
- Dump SQL MySQL/Postgres hanya didukung seadanya (dialek SQLite). Untuk yang rumit, ekspor ke CSV.
- Format ribuan gaya Indonesia (`1.200`) tidak ditebak saat membersihkan angka-teks.
- Gambar dianalisis lewat fitur visual dasar + deskripsi AI, belum klasifikasi/deteksi objek.
- File `.xls` lama memerlukan paket `xlrd`.

Lihat [docs/ROADMAP.md](docs/ROADMAP.md) untuk rencana berikutnya.

## Lisensi
MIT
