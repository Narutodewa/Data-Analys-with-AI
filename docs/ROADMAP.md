# Roadmap

Fitur tambahan yang disarankan karena aplikasi ini berbasis AI. Urutan = perkiraan prioritas.

## Berikutnya (v0.2)
- [ ] **Sesi & penyimpanan**: simpan proyek (dataset + riwayat cleaning) ke disk/SQLite; multi-pengguna dengan autentikasi.
- [ ] **Dashboard**: simpan grafik ke kanvas, susun, filter lintas grafik (ala Tableau), ekspor HTML/PDF.
- [ ] **Laporan otomatis**: AI menulis ringkasan temuan + grafik terpilih menjadi laporan.
- [ ] **Resep cleaning yang bisa diputar ulang**: ekspor langkah cleaning sebagai skrip Python/SQL dan terapkan ke file baru.
- [ ] **Join/merge & append** antar dataset; pivot dengan banyak nilai.

## Analisis lanjutan (v0.3)
- [ ] Uji statistik (t-test, ANOVA, chi-square) dengan penjelasan awam dan peringatan asumsi.
- [ ] **Deteksi anomali** (Isolation Forest) dan **clustering** (k-means/HDBSCAN) dengan penjelasan segmen oleh AI.
- [ ] **Prediksi cepat**: baseline regresi/klasifikasi + feature importance; peramalan deret waktu.
- [ ] **AI imputation**: pengisian nilai kosong berbasis model (KNN/iterative) sebagai opsi di Cleaning.
- [ ] Deteksi **kebocoran data**, bias, dan ketidakseimbangan kelas.

## Data tak terstruktur
- [ ] Koneksi langsung **MongoDB/PostgreSQL/MySQL** (baca saja) selain file.
- [ ] Citra: klasifikasi/deteksi objek (CLIP/YOLO), deteksi duplikat/near-duplicate, pelabelan.
- [ ] Teks: pembersihan, frekuensi kata, topik, sentimen, dan ekstraksi entitas via AI.
- [ ] PDF/dokumen: ekstraksi tabel.

## Kualitas & operasional
- [ ] Streaming/chunking untuk file > RAM; DuckDB sebagai mesin kueri.
- [ ] Mode gelap grafik yang tervalidasi penuh, tabel alternatif aksesibilitas per grafik.
- [ ] Tes UI end-to-end (Playwright) di CI.
- [ ] i18n (Indonesia/Inggris).
