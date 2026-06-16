# ============================================================
# SOAL 11.1 - Thomas Algorithm untuk Sistem Tridiagonal
# Nama : GRACIA MARSELINA
# NIM  : F5512510024
# ============================================================
# Sistem:
# [ 0.8  -0.4        ] [x1]   [ 41 ]
# [-0.4   0.8  -0.4  ] [x2] = [ 25 ]
# [       -0.4  0.8  ] [x3]   [105 ]
# ============================================================

import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Algoritma Thomas untuk sistem tridiagonal.
    a = subdiagonal, b = diagonal utama, c = superdiagonal, d = RHS
    """
    n = len(d)
    # Salin array supaya tidak mengubah input asli
    c_ = c.copy().astype(float)
    d_ = d.copy().astype(float)
    b_ = b.copy().astype(float)

    # Forward sweep
    for i in range(1, n):
        m = a[i] / b_[i-1]
        b_[i] -= m * c_[i-1]
        d_[i] -= m * d_[i-1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d_[-1] / b_[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d_[i] - c_[i] * x[i+1]) / b_[i]

    return x

# ── (a) Soal 11.1 ──────────────────────────────────────────
print("=" * 50)
print("SOAL 11.1 – Thomas Algorithm (Tridiagonal)")
print("=" * 50)

# Diagonal utama
b = np.array([0.8, 0.8, 0.8])
# Subdiagonal (a[0] tidak dipakai)
a = np.array([0.0, -0.4, -0.4])
# Superdiagonal (c[-1] tidak dipakai)
c = np.array([-0.4, -0.4, 0.0])
# RHS
d = np.array([41.0, 25.0, 105.0])

x = thomas_algorithm(a, b, c, d)

print(f"\n(a) Solusi sistem tridiagonal:")
for i, xi in enumerate(x):
    print(f"   x{i+1} = {xi:.6f}")

# Verifikasi dengan numpy
A = np.array([[0.8, -0.4, 0.0],
              [-0.4, 0.8, -0.4],
              [0.0, -0.4, 0.8]])
rhs = np.array([41.0, 25.0, 105.0])
x_np = np.linalg.solve(A, rhs)
print(f"\n   Verifikasi (numpy.linalg.solve):")
for i, xi in enumerate(x_np):
    print(f"   x{i+1} = {xi:.6f}")

# ── (b) Soal 11.3 ──────────────────────────────────────────
print("\n(b) Soal 11.3 – Crank-Nicolson Tridiagonal:")
b2 = np.array([2.01475, 2.01475, 2.01475, 2.01475])
a2 = np.array([0.0, -0.020875, -0.020875, -0.020875])
c2 = np.array([-0.020875, -0.020875, -0.020875, 0.0])
d2 = np.array([4.175, 0.0, 0.0, 2.0875])

T = thomas_algorithm(a2, b2, c2, d2)
for i, ti in enumerate(T):
    print(f"   T{i+1} = {ti:.6f}")
