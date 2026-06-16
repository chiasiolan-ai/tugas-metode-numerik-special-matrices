# ============================================================
# SOAL 11.28 - Pentadiagonal System Solver (bandwidth=5)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 55)
print("SOAL 11.28 – Pentadiagonal System Solver")
print("=" * 55)

def pentadiag_solve(d_sub2, d_sub1, diag, d_sup1, d_sup2, rhs):
    """
    Solver untuk sistem pentadiagonal (bandwidth=5).
    Tanpa pivoting — mirip Thomas algorithm tapi untuk bandwidth 5.

    Band:
    d_sub2 : subdiagonal ke-2  (e)
    d_sub1 : subdiagonal ke-1  (f bawah → d)
    diag   : diagonal utama    (f)
    d_sup1 : superdiagonal ke-1 (g)
    d_sup2 : superdiagonal ke-2 (h)
    rhs    : sisi kanan
    """
    n = len(rhs)
    # Salin semua agar tidak ubah array asli
    e  = np.array(d_sub2, dtype=float)
    d  = np.array(d_sub1, dtype=float)
    f  = np.array(diag,   dtype=float)
    g  = np.array(d_sup1, dtype=float)
    h  = np.array(d_sup2, dtype=float)
    r  = np.array(rhs,    dtype=float)

    # Forward elimination
    for i in range(1, n):
        if abs(f[i-1]) < 1e-15:
            raise ZeroDivisionError(f"Pivot nol pada i={i-1}")

        # Eliminasi dengan baris i-1
        fac1 = d[i] / f[i-1]
        f[i] -= fac1 * g[i-1]
        g[i] -= fac1 * h[i-1]
        r[i] -= fac1 * r[i-1]
        d[i]  = 0.0

        # Eliminasi dengan baris i-2 (jika ada)
        if i >= 2:
            fac2 = e[i] / f[i-2]
            f[i] -= fac2 * g[i-2]   # koreksi melalui baris i-2
            r[i] -= fac2 * r[i-2]
            e[i]  = 0.0

    # Back substitution
    x = np.zeros(n)
    x[-1] = r[-1] / f[-1]
    if n > 1:
        x[-2] = (r[-2] - g[-2]*x[-1]) / f[-2]
    for i in range(n-3, -1, -1):
        x[i] = (r[i] - g[i]*x[i+1] - h[i]*x[i+2]) / f[i]

    return x


# ── Test sistem 5×5 dari soal ────────────────────────────────
print("""
Sistem:
[ 8  -2  -1   0   0] [x1]   [5]
[-2   9  -4  -1   0] [x2]   [2]
[-1  -3   7  -1  -2] [x3] = [0]
[ 0  -4  -2  12  -5] [x4]   [1]
[ 0   0  -7  -3  15] [x5]   [5]
""")

A_full = np.array([[ 8, -2, -1,  0,  0],
                   [-2,  9, -4, -1,  0],
                   [-1, -3,  7, -1, -2],
                   [ 0, -4, -2, 12, -5],
                   [ 0,  0, -7, -3, 15]], dtype=float)
rhs = np.array([5, 2, 0, 1, 5], dtype=float)

# Ekstrak band
n = 5
diag   = np.diag(A_full, 0).copy()
d_sup1 = np.concatenate([np.diag(A_full, 1), [0]])
d_sup2 = np.concatenate([np.diag(A_full, 2), [0, 0]])
d_sub1 = np.concatenate([[0], np.diag(A_full, -1)])
d_sub2 = np.concatenate([[0, 0], np.diag(A_full, -2)])

x = pentadiag_solve(d_sub2, d_sub1, diag, d_sup1, d_sup2, rhs)

print("Solusi (Pentadiagonal Solver):")
for i, xi in enumerate(x):
    print(f"   x{i+1} = {xi:.6f}")

x_np = np.linalg.solve(A_full, rhs)
print("\nVerifikasi (numpy.linalg.solve):")
for i, xi in enumerate(x_np):
    print(f"   x{i+1} = {xi:.6f}")

print(f"\nMaks error vs numpy: {np.max(np.abs(x-x_np)):.2e}")
