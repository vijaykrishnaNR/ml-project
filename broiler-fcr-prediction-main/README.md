[README_Proyek1_FCR_Prediction.md](https://github.com/user-attachments/files/26595907/README_Proyek1_FCR_Prediction.md)
# 🐔 Broiler FCR Prediction Model

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Poultry%20%7C%20AgriTech-yellow)

> **Memprediksi Feed Conversion Ratio (FCR) ayam broiler secara proaktif berdasarkan data harian kandang — membantu peternak mengambil tindakan sebelum efisiensi pakan memburuk.**

---

## 📌 Latar Belakang

**Feed Conversion Ratio (FCR)** adalah indikator terpenting dalam peternakan broiler komersial. FCR mengukur seberapa efisien ayam mengkonversi pakan menjadi bobot badan:

```
FCR = Total Pakan yang Dikonsumsi (g) / Total Pertambahan Bobot Badan (g)
```

- FCR **rendah** (1.4–1.6) = efisien, biaya produksi rendah ✅  
- FCR **tinggi** (>1.9) = boros pakan, margin menyempit ⚠️

Masalahnya: peternak biasanya baru menyadari FCR memburuk **setelah** efisiensi turun — bukan sebelumnya. Proyek ini membangun model prediktif yang memberikan **early warning** agar peternak bisa intervensi lebih awal.

---

## 🎯 Tujuan Proyek

1. Membangun model ML yang memprediksi FCR berdasarkan variabel kandang harian
2. Mengidentifikasi faktor-faktor paling berpengaruh terhadap FCR
3. Membuat sistem alert sederhana berdasarkan prediksi FCR (Efisien / Perhatian / Kritis)
4. Mengevaluasi dan menangani limitasi model (extrapolation problem)

---

## 📊 Dataset

| Sumber | Keterangan |
|--------|-----------|
| **Base Dataset** | [ChickWeight — Chicken Weight vs Feed](https://www.kaggle.com/datasets/yamqwe/chicken-weight-vs-different-chicken-feeds) (Kaggle) |
| **Pendekatan** | Hybrid: dataset nyata sebagai fondasi + synthetic features berbasis literatur broiler tropis (Ross 308, PMC/MDPI 2023) |
| **Total Data** | 540 observasi |
| **Fitur** | 9 variabel (umur, bobot, konsumsi pakan, suhu, kelembaban, kepadatan, mortalitas, jenis pakan) |
| **Target** | FCR (kontinu, range 1.35–2.10) |

**Kenapa Hybrid?** Dataset publik FCR broiler yang mencakup variabel lingkungan kandang (suhu, kepadatan, kelembaban) tidak tersedia secara publik dalam format ML-ready. Pendekatan hybrid dengan synthetic features berbasis literatur ilmiah adalah praktik umum dan valid dalam penelitian ML peternakan.

---

## 🔍 Exploratory Data Analysis

### Korelasi Fitur terhadap FCR

| Fitur | Korelasi | Interpretasi |
|-------|----------|--------------|
| `umur_hari` | +0.605 | Semakin tua, FCR cenderung naik |
| `konsumsi_pakan_g` | +0.560 | Konsumsi tinggi relatif bobot = FCR buruk |
| `bobot_badan_g` | +0.502 | Bobot tinggi = FCR lebih efisien |
| `suhu_kandang_C` | -0.546 | Suhu tinggi = makan lebih sedikit → FCR turun sementara |
| `kelembaban_pct` | -0.147 | Pengaruh lemah |
| `mortalitas_pct` | +0.011 | Tidak signifikan |

### Visualisasi

<table>
<tr>
<td><img src="images/fcr_distribution.png" width="380"/><br><sub>Distribusi FCR (target)</sub></td>
<td><img src="images/correlation_heatmap.png" width="380"/><br><sub>Heatmap korelasi antar fitur</sub></td>
</tr>
<tr>
<td><img src="images/feature_vs_fcr.png" width="380"/><br><sub>Scatter plot fitur utama vs FCR</sub></td>
<td><img src="images/feature_importance.png" width="380"/><br><sub>Feature importance (koefisien model)</sub></td>
</tr>
</table>

---

## 🤖 Modeling

Tiga model dibandingkan untuk menemukan yang terbaik:

| Model | MAE | RMSE | R² Score | CV R² (5-fold) |
|-------|-----|------|----------|----------------|
| **Linear Regression** ⭐ | **0.0482** | **0.0597** | **0.7072** | **0.7205** |
| Gradient Boosting | 0.0525 | 0.0659 | 0.6430 | 0.5499 |
| Random Forest | 0.0559 | 0.0703 | 0.5928 | 0.4767 |

**Linear Regression terpilih** karena menghasilkan R² tertinggi dan CV R² konsisten — menunjukkan tidak ada overfitting.

### Interpretasi Koefisien (Feature Importance)

```
konsumsi_pakan_g  (+0.65) → Prediktor TERKUAT — rasio konsumsi terhadap bobot = FCR
bobot_badan_g     (-0.55) → Bobot tinggi justru turunkan FCR (pertumbuhan optimal = efisien)
umur_hari         (+0.08) → Pengaruh kecil tapi nyata, FCR naik seiring umur
suhu & kelembaban (~0.00) → Efeknya sudah ter-capture melalui perubahan konsumsi pakan
```

---

## ⚠️ Limitasi & Penanganan

### Masalah: Extrapolation Problem
Model Linear Regression memprediksi nilai di luar batas biologis (FCR = 4.15) ketika input berada jauh di luar range training (suhu >33°C + konsumsi sangat tinggi).

### Solusi yang Diimplementasikan

**Opsi 1 — Output Clipping:**
```python
y_pred_fixed = np.clip(y_pred, FCR_MIN=1.35, FCR_MAX=2.10)
```
Mencegah prediksi tidak realistis. Tidak mengubah metrik pada data normal (+0.00% perubahan MAE), karena test data sudah dalam range normal — ini membuktikan model stabil dalam kondisi biasa.

**Opsi 2 — Gradient Boosting dengan Clipped Target:**
Dilatih ulang dengan target yang di-clip, namun menghasilkan R² = 0.6430 (lebih rendah dari LR). Kesimpulan: untuk data synthetic linear ini, LR + Clipping adalah kombinasi terbaik.

---

## 📈 Hasil Akhir & Demo Prediksi

```
=======================================================
  RINGKASAN FINAL PROYEK 1
=======================================================

  Model          : Linear Regression + Output Clipping
  R² Score       : 0.7072
  MAE            : 0.0482
  CV R² (5-fold) : 0.7205

  Demo Prediksi 5 Kandang:
  ┌─────────────────────────────────────────────────┐
  │ Umur  Suhu   FCR Prediksi   Status              │
  │  21   27°C   1.357          ✅ Efisien           │
  │  28   28°C   1.410          ✅ Efisien           │
  │  35   29°C   2.100          🚨 Kritis (clipped)  │
  │  21   33°C   2.100          🚨 Kritis (clipped)  │
  │  28   34°C   2.100          🚨 Kritis (clipped)  │
  └─────────────────────────────────────────────────┘
```

<img src="images/actual_vs_predicted.png" width="600"/><br>
<sub>Actual vs Predicted FCR — titik-titik mengikuti garis ideal (R²=0.71)</sub>

---

## 💡 Rekomendasi Praktis

- **Pantau rasio konsumsi pakan vs bobot badan** harian sebagai early warning indicator
- **Panen optimal di hari 21–28** sebelum FCR mulai naik signifikan
- **Pertahankan pertumbuhan bobot badan yang optimal** — bobot tinggi berarti FCR lebih efisien
- **Untuk deployment produksi**, latih ulang model dengan data kandang nyata agar akurasi dan generalisasi lebih baik

---

## 🛠️ Tech Stack

```
Python 3.10+
├── pandas & numpy        → manipulasi data
├── matplotlib & seaborn  → visualisasi
├── scikit-learn          → modeling & evaluasi
└── statsmodels           → load base dataset
```

## ▶️ Cara Menjalankan

1. Buka di [Kaggle Notebook](https://kaggle.com) atau Google Colab
2. Tambahkan dataset: [Chicken Weight vs Feed](https://www.kaggle.com/datasets/yamqwe/chicken-weight-vs-different-chicken-feeds)
3. Jalankan `notebook.ipynb` sel per sel dari atas ke bawah

---

## 📁 Struktur Repo

```
broiler-fcr-prediction/
├── notebook.ipynb          ← notebook lengkap (semua tahap)
├── README.md
├── requirements.txt
└── images/
    ├── fcr_distribution.png
    ├── correlation_heatmap.png
    ├── feature_vs_fcr.png
    ├── feature_importance.png
    ├── actual_vs_predicted.png
    └── extrapolation_fix.png
```

---

## 🔗 Konteks Bisnis

Proyek ini dibuat sebagai bagian dari persiapan lamaran posisi **AI/ML Innovation Intern** di [Chickin Indonesia](https://chickin.id) — startup agritech yang membangun ekosistem peternakan broiler terintegrasi teknologi untuk 50.000+ mitra peternak di Indonesia.

Model ini dirancang dengan tujuan akhir diintegrasikan ke platform **Chickin Smartfarm** untuk menyediakan prediksi FCR real-time berbasis data sensor IoT kandang.

---

*Dibuat dengan kombinasi domain knowledge peternakan + praktik ML industri*
