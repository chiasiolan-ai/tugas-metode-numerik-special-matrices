# ============================================================
# SOAL 11.7 - Cholesky Decomposition Diagonal Matrix
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.7 – Cholesky Diagonal Matrix")
print("=" * 50)

A = np.array([[900, 0, 0],
              [0, 25, 0],
              [0, 0, 4]], dtype=float)

print("\nMatriks A (diagonal):"); print(A)

def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for k in range(i+1):
            s = sum(L[i,j]*L[k,j] for j in range(k))
            L[i,k] = np.sqrt(A[i,i]-s) if i==k else (A[i,k]-s)/L[k,k]
    return L

L = cholesky(A)
print("\nMatriks L:")
print(np.round(L,6))
print("\nInterpretasi: L diagonal = sqrt dari elemen diagonal A")
print(f"   L[0,0] = sqrt(900) = {np.sqrt(900):.2f}")
print(f"   L[1,1] = sqrt(25)  = {np.sqrt(25):.2f}")
print(f"   L[2,2] = sqrt(4)   = {np.sqrt(4):.2f}")
print("\nVerifikasi L*L^T = A:"); print(np.round(L@L.T,4))
