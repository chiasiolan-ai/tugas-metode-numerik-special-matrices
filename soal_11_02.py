# ============================================================
# SOAL 11.2 - Invers Matriks via Dekomposisi LU
# Nama : GRACIA MARSELINA
# NIM  : F5512510024
# ============================================================

import numpy as np
from scipy.linalg import lu

print("=" * 50)
print("SOAL 11.2 – Invers Matriks via LU Decomposition")
print("=" * 50)

# Matriks dari Example 11.1 (tridiagonal)
A = np.array([[0.8, -0.4, 0.0],
              [-0.4, 0.8, -0.4],
              [0.0, -0.4, 0.8]], dtype=float)

print("\nMatriks A:")
print(A)

# Dekomposisi LU
P, L, U = lu(A)
print("\nMatriks L (Lower):")
print(np.round(L, 6))
print("\nMatriks U (Upper):")
print(np.round(U, 6))

# Hitung invers menggunakan unit vectors
n = A.shape[0]
A_inv = np.zeros((n, n))
for i in range(n):
    e = np.zeros(n)
    e[i] = 1.0
    # Solve A * col_i = e_i
    A_inv[:, i] = np.linalg.solve(A, e)

print("\nInvers Matriks A (via unit vectors):")
print(np.round(A_inv, 6))

# Verifikasi: A * A_inv = I
I_check = np.round(A @ A_inv, 8)
print("\nVerifikasi A * A_inv (harus = I):")
print(I_check)
