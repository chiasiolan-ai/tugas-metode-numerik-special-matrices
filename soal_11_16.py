# ============================================================
# SOAL 11.16 - Solusi, Invers, Condition Number (Row-Sum Norm)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.16 – Solusi, Invers & Condition Number")
print("=" * 50)

# (a) Sistem 3x3
print("\n(a) Sistem 3×3:")
A3 = np.array([[1, 4, 9],
               [4, 9, 16],
               [9, 16, 25]], dtype=float)
b3 = np.array([14.0, 29.0, 50.0])

x3 = np.linalg.solve(A3, b3)
print(f"   Solusi: x1={x3[0]:.6f}, x2={x3[1]:.6f}, x3={x3[2]:.6f}")
print(f"   (Seharusnya semua = 1.0)")

A3_inv = np.linalg.inv(A3)
print(f"   Invers A:\n{np.round(A3_inv,4)}")

# Row-sum norm = max row sum of |A|
norm_A   = np.max(np.sum(np.abs(A3), axis=1))
norm_Ai  = np.max(np.sum(np.abs(A3_inv), axis=1))
cond_row = norm_A * norm_Ai
cond_np  = np.linalg.cond(A3, np.inf)
print(f"   Condition number (row-sum): {cond_row:.4f}")
print(f"   Condition number (numpy):   {cond_np:.4f}")

# (b) Sistem 4x4
print("\n(b) Sistem 4×4:")
A4 = np.array([[1,4,9,16],
               [4,9,16,25],
               [9,16,25,36],
               [16,25,36,49]], dtype=float)
b4 = np.array([30.0, 54.0, 86.0, 126.0])

x4 = np.linalg.solve(A4, b4)
print(f"   Solusi: {np.round(x4,6)}")
print(f"   (Seharusnya semua = 1.0)")

A4_inv = np.linalg.inv(A4)
norm_A4  = np.max(np.sum(np.abs(A4), axis=1))
norm_Ai4 = np.max(np.sum(np.abs(A4_inv), axis=1))
cond4    = norm_A4 * norm_Ai4
print(f"   Condition number (row-sum): {cond4:.4f}")
print(f"   Condition number (numpy):   {np.linalg.cond(A4, np.inf):.4f}")
