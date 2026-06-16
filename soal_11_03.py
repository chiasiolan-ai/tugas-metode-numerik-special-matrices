# ============================================================
# SOAL 11.3 - Thomas Algorithm: Crank-Nicolson PDE
# Nama : GRACIA MARSELINA
# NIM  : F5512510024
# ============================================================

import numpy as np

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_ = c.copy().astype(float)
    d_ = d.copy().astype(float)
    b_ = b.copy().astype(float)
    for i in range(1, n):
        m = a[i] / b_[i-1]
        b_[i] -= m * c_[i-1]
        d_[i] -= m * d_[i-1]
    x = np.zeros(n)
    x[-1] = d_[-1] / b_[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d_[i] - c_[i] * x[i+1]) / b_[i]
    return x

print("=" * 50)
print("SOAL 11.3 – Crank-Nicolson Tridiagonal (Thomas)")
print("=" * 50)

# Sistem:
# [ 2.01475  -0.020875                        ] [T1]   [4.175 ]
# [-0.020875  2.01475  -0.020875              ] [T2] = [0     ]
# [          -0.020875  2.01475  -0.020875    ] [T3]   [0     ]
# [                    -0.020875  2.01475     ] [T4]   [2.0875]

b = np.array([2.01475, 2.01475, 2.01475, 2.01475])
a = np.array([0.0, -0.020875, -0.020875, -0.020875])
c = np.array([-0.020875, -0.020875, -0.020875, 0.0])
d = np.array([4.175, 0.0, 0.0, 2.0875])

T = thomas_algorithm(a, b, c, d)

print("\nSolusi (Thomas Algorithm):")
for i, ti in enumerate(T):
    print(f"   T{i+1} = {ti:.6f}")

# Verifikasi
A = np.array([
    [2.01475, -0.020875, 0, 0],
    [-0.020875, 2.01475, -0.020875, 0],
    [0, -0.020875, 2.01475, -0.020875],
    [0, 0, -0.020875, 2.01475]
])
rhs = np.array([4.175, 0.0, 0.0, 2.0875])
T_np = np.linalg.solve(A, rhs)
print("\nVerifikasi (numpy):")
for i, ti in enumerate(T_np):
    print(f"   T{i+1} = {ti:.6f}")
