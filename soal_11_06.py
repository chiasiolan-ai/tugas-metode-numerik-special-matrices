# ============================================================
# SOAL 11.6 - Cholesky Decomposition Manual
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.6 – Cholesky Decomposition (Manual)")
print("=" * 50)

A = np.array([[8, 20, 15],
              [20, 80, 50],
              [15, 50, 60]], dtype=float)
b = np.array([50.0, 250.0, 100.0])

def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for k in range(i+1):
            s = sum(L[i,j]*L[k,j] for j in range(k))
            L[i,k] = np.sqrt(A[i,i]-s) if i==k else (A[i,k]-s)/L[k,k]
    return L

L = cholesky(A)
print("\nMatriks L:"); print(np.round(L,6))
print("\nVerifikasi L*L^T:"); print(np.round(L@L.T,4))

# Solve
n = len(b)
y = np.zeros(n)
for i in range(n):
    y[i] = (b[i] - sum(L[i,j]*y[j] for j in range(i)))/L[i,i]
x = np.zeros(n)
LT = L.T
for i in range(n-1,-1,-1):
    x[i] = (y[i] - sum(LT[i,j]*x[j] for j in range(i+1,n)))/LT[i,i]
print("\nSolusi:")
for i,xi in enumerate(x): print(f"   x{i+1} = {xi:.6f}")
