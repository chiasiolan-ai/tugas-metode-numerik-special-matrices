# ============================================================
# SOAL 11.5 - Cholesky Decomposition & Solve
# Nama : GRACIA MARSELINA
# NIM  : F5512510024
# ============================================================

import numpy as np

print("=" * 50)
print("SOAL 11.5 – Cholesky Decomposition & Solve")
print("=" * 50)

A = np.array([[6, 15, 55],
              [15, 55, 225],
              [55, 225, 979]], dtype=float)

b = np.array([152.6, 585.6, 2488.8])

print("\nMatriks A:")
print(A)
print(f"\nVector b: {b}")

# Dekomposisi Cholesky
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
print("\nMatriks L:")
print(np.round(L, 6))

# Forward substitution: L * y = b
n = len(b)
y = np.zeros(n)
for i in range(n):
    y[i] = (b[i] - sum(L[i,j]*y[j] for j in range(i))) / L[i,i]

# Back substitution: L^T * x = y
x = np.zeros(n)
LT = L.T
for i in range(n-1, -1, -1):
    x[i] = (y[i] - sum(LT[i,j]*x[j] for j in range(i+1, n))) / LT[i,i]

print("\nSolusi:")
labels = ['a0', 'a1', 'a2']
for label, xi in zip(labels, x):
    print(f"   {label} = {xi:.6f}")

print("\nVerifikasi (numpy.linalg.solve):")
x_np = np.linalg.solve(A, b)
for label, xi in zip(labels, x_np):
    print(f"   {label} = {xi:.6f}")
