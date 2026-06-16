# ============================================================
# SOAL 11.25 - Program Cholesky Decomposition (User-Friendly)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

def cholesky_decompose(A, verbose=True):
    """
    Cholesky Decomposition: A = L * L^T
    Hanya berlaku untuk matriks simetris positif definit.
    """
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    if verbose:
        print("\n--- Proses Dekomposisi Cholesky ---")

    for i in range(n):
        for k in range(i+1):
            s = sum(L[i,j] * L[k,j] for j in range(k))
            if i == k:
                val = A[i,i] - s
                if val < 0:
                    raise ValueError(f"Matriks tidak positif definit! A[{i},{i}]-sum = {val:.4f}")
                L[i,k] = np.sqrt(val)
            else:
                L[i,k] = (A[i,k] - s) / L[k,k]
            if verbose:
                print(f"   L[{i},{k}] = {L[i,k]:.6f}")
    return L

def cholesky_solve(L, b, verbose=True):
    """Solve A*x = b given L from Cholesky (A = L*L^T)."""
    n = len(b)
    # Forward: L * y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i,j]*y[j] for j in range(i))) / L[i,i]

    # Backward: L^T * x = y
    x = np.zeros(n)
    LT = L.T
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(LT[i,j]*x[j] for j in range(i+1,n))) / LT[i,i]

    if verbose:
        print(f"\n   y (forward sub) = {np.round(y,6)}")
        print(f"   x (back sub)    = {np.round(x,6)}")
    return x

# ── Test dengan Example 11.2 ────────────────────────────────
print("=" * 55)
print("SOAL 11.25 – Program Cholesky Decomposition")
print("=" * 55)
print("\nTest: Sistem dari Example 11.2")

A = np.array([[6, 15, 55],
              [15, 55, 225],
              [55, 225, 979]], dtype=float)
b = np.array([152.6, 585.6, 2488.8])

print("\nMatriks A:"); print(A)
print(f"Vector b: {b}")

L = cholesky_decompose(A, verbose=True)
print("\nMatriks L:"); print(np.round(L,6))
print("\nVerifikasi L*L^T:"); print(np.round(L@L.T,4))

x = cholesky_solve(L, b, verbose=True)
print(f"\nSolusi:")
labels = ['a0','a1','a2']
for label,xi in zip(labels,x):
    print(f"   {label} = {xi:.6f}")

print("\nVerifikasi A*x ≈ b:")
print(f"   {np.round(A@x,4)} ≈ {b}")
