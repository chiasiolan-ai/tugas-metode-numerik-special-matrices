# ============================================================
# SOAL 11.4 - Verifikasi Dekomposisi Cholesky
# Nama : GRACIA MARSELINA
# NIM  : F5512510024
# ============================================================

import numpy as np

print("=" * 50)
print("SOAL 11.4 – Verifikasi Cholesky: L * L^T = A")
print("=" * 50)

# Matriks A dari Example 11.2 (simetris positif definit)
A = np.array([[6, 15, 55],
              [15, 55, 225],
              [55, 225, 979]], dtype=float)

print("\nMatriks A:")
print(A)

# Dekomposisi Cholesky manual
def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for k in range(i+1):
            s = sum(L[i,j] * L[k,j] for j in range(k))
            if i == k:
                L[i,k] = np.sqrt(A[i,i] - s)
            else:
                L[i,k] = (A[i,k] - s) / L[k,k]
    return L

L = cholesky(A)
print("\nMatriks L (Lower Triangular):")
print(np.round(L, 6))

print("\nMatriks L^T (Transpose):")
print(np.round(L.T, 6))

# Verifikasi L * L^T = A
A_rekonstruksi = L @ L.T
print("\nL * L^T (harus = A):")
print(np.round(A_rekonstruksi, 6))

error = np.max(np.abs(A - A_rekonstruksi))
print(f"\nMaks error = {error:.2e}  → {'✅ VALID' if error < 1e-8 else '❌ TIDAK VALID'}")
