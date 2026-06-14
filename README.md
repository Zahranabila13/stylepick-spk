# 👗 StylePick SPK — SAW & TOPSIS

Aplikasi Sistem Pendukung Keputusan pemilihan baju menggunakan metode **SAW** dan **TOPSIS**.

## Cara Menjalankan

### 1. Install Python (jika belum)
Download Python 3.10+ dari https://python.org

### 2. Install dependencies
Buka terminal/command prompt di folder ini, lalu jalankan:
```
pip install -r requirements.txt
```

### 3. Jalankan aplikasi
```
streamlit run stylepick_spk.py
```

Aplikasi akan terbuka otomatis di browser: http://localhost:8501

---

## Fitur Aplikasi

| Tab | Fitur |
|-----|-------|
| 📊 Data & Kriteria | Edit nama baju, nilai semua kriteria, tambah/hapus baju & kriteria |
| ⚖️ Bobot Kriteria | Slider bobot real-time + grafik donut & stacked bar |
| ⚡ Metode SAW | Perhitungan step-by-step + contoh manual + grafik |
| 🎯 Metode TOPSIS | Perhitungan step-by-step + A+/A− + D+/D− + grafik |
| 🏆 Hasil Akhir | Radar chart, ranking SAW & TOPSIS, tabel perbandingan, kesimpulan |

## Deploy ke Internet (Gratis)

Bisa deploy ke **Streamlit Community Cloud**:
1. Upload kode ke GitHub
2. Buka https://share.streamlit.io
3. Connect repo → Deploy

---

Dibuat untuk UAS Mata Kuliah Sistem Pengambilan Keputusan
