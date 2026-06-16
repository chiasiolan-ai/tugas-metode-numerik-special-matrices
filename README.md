# Tugas Metode Numerik — Special Matrices & Gauss-Seidel

| | |
|---|---|
| **Nama** | GRACIA MARSELINA |
| **NIM** | F5512510024 |
| **Mata Kuliah** | Metode Numerik |
| **Topik** | Special Matrices & Gauss-Seidel (Bab 11) |
| **Sumber** | Numerical Methods for Engineers — Chapra & Canale |

---

## Library yang Digunakan

| Library | Versi | Kegunaan |
|---------|-------|---------|
| `numpy` | ≥1.21 | Operasi matriks, aljabar linear, array |
| `scipy` | ≥1.7  | Dekomposisi LU (`scipy.linalg.lu`), solver (`fsolve`) |
| `matplotlib` | ≥3.4 | Visualisasi grafik & plot |

### Instalasi
```bash
pip install numpy scipy matplotlib
```

---

## Daftar Soal & File

| File | Soal | Topik |
|------|------|-------|
| `soal_11_01.py` | 11.1 | Thomas Algorithm — Sistem Tridiagonal |
| `soal_11_02.py` | 11.2 | Invers Matriks via Dekomposisi LU |
| `soal_11_03.py` | 11.3 | Thomas Algorithm — Crank-Nicolson PDE |
| `soal_11_04.py` | 11.4 | Verifikasi Dekomposisi Cholesky (L·Lᵀ = A) |
| `soal_11_05.py` | 11.5 | Cholesky Decomposition & Solve |
| `soal_11_06.py` | 11.6 | Cholesky Manual — Sistem Simetris |
| `soal_11_07.py` | 11.7 | Cholesky — Matriks Diagonal |
| `soal_11_08.py` | 11.8 | Gauss-Seidel + Overrelaxation (λ=1.2) |
| `soal_11_09.py` | 11.9 | Gauss-Seidel — Reaktor Kimia |
| `soal_11_10.py` | 11.10 | Jacobi Iteration — Reaktor Kimia |
| `soal_11_11.py` | 11.11 | Gauss-Seidel (es < 5%) |
| `soal_11_12.py` | 11.12 | Gauss-Seidel ± Relaxasi (λ=0.95) |
| `soal_11_13.py` | 11.13 | Gauss-Seidel ± Relaxasi (λ=1.2) |
| `soal_11_14.py` | 11.14 | Analisis Konvergensi Slope 1 & -1 |
| `soal_11_15.py` | 11.15 | Identifikasi Set Konvergen/Divergen |
| `soal_11_16.py` | 11.16 | Solusi, Invers & Condition Number |
| `soal_11_17.py` | 11.17 | Sistem Nonlinear (Newton-Raphson) |
| `soal_11_18.py` | 11.18 | Perencanaan Produksi Elektronik |
| `soal_11_19.py` | 11.19 | Hilbert Matrix 10×10 |
| `soal_11_20.py` | 11.20 | Vandermonde Matrix 6×6 |
| `soal_11_21.py` | 11.21 | Augmented Matrix [A \| I] |
| `soal_11_22.py` | 11.22 | Sistem Persamaan → Bentuk Matriks |
| `soal_11_23.py` | 11.23 | Perbandingan Operasi Thomas vs Gauss |
| `soal_11_24.py` | 11.24 | Program Thomas Algorithm |
| `soal_11_25.py` | 11.25 | Program Cholesky Decomposition |
| `soal_11_26.py` | 11.26 | Program Gauss-Seidel |
| `soal_11_27.py` | 11.27 | PDE Kanal 1D (Finite Difference) |
| `soal_11_28.py` | 11.28 | Pentadiagonal System Solver |

---

## Cara Menjalankan

```bash
# Jalankan satu soal
python soal_11_01.py

# Jalankan semua sekaligus
for f in soal_11_*.py; do echo "=== $f ==="; python "$f"; done
```

---

## Referensi

Chapra, S.C. & Canale, R.P. (2015). *Numerical Methods for Engineers*, 7th Edition. McGraw-Hill.
