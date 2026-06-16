# ============================================================
# SOAL 11.22 - Sistem Persamaan dalam Bentuk Matriks
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.22 – Bentuk Matriks & Solve")
print("=" * 50)
print("""
Persamaan asli:
   50 = 5x3 - 7x2
   4x2 + 7x3 + 30 = 0
   x1 - 7x3 = 40 - 3x2 + 5x1

Susun ulang:
   -7x2 + 5x3        = 50
    4x2 + 7x3        = -30
   -4x1 + 3x2 - 7x3  = 40  → (x1-7x3=40-3x2+5x1 → -4x1+3x2-7x3=40)
""")

# Koefisien [x1, x2, x3]
A = np.array([[0, -7, 5],
              [0,  4, 7],
              [-4, 3, -7]], dtype=float)
b = np.array([50.0, -30.0, 40.0])

print("Matriks A:")
print(A)
print(f"\nVector b: {b}")

x = np.linalg.solve(A, b)
print(f"\nSolusi:")
print(f"   x1 = {x[0]:.6f}")
print(f"   x2 = {x[1]:.6f}")
print(f"   x3 = {x[2]:.6f}")

# Transpose dan invers
A_T = A.T
A_inv = np.linalg.inv(A)

print("\nTranspose A^T:")
print(np.round(A_T, 4))

print("\nInvers A^(-1):")
print(np.round(A_inv, 4))

print("\nVerifikasi A * A^(-1) = I:")
print(np.round(A @ A_inv, 6))
