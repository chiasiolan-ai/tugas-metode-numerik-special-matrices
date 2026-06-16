# ============================================================
# SOAL 11.20 - Vandermonde Matrix 6x6: Condition Number
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.20 – Vandermonde Matrix 6x6")
print("=" * 50)

# x1=4, x2=2, x3=7, x4=10, x5=3, x6=5
xv = np.array([4, 2, 7, 10, 3, 5], dtype=float)
n = len(xv)

# Vandermonde: V[i,j] = x[i]^(n-1-j)
V = np.vander(xv, increasing=False)
print("\nVandermonde Matrix (3 baris pertama):")
print(V[:3])

cond = np.linalg.cond(V)
print(f"\nCondition Number (spektral): {cond:.4e}")
digits_lost = np.log10(cond)
print(f"Perkiraan digit presisi hilang: {digits_lost:.1f} digit")

b = V.sum(axis=1)
x = np.linalg.solve(V, b)

print(f"\nSolusi (seharusnya semua = 1):")
for i in range(n):
    print(f"   x{i+1} = {x[i]:>12.6f}  error = {abs(x[i]-1):.2e}")

print(f"\nMaks error aktual : {np.max(np.abs(x-1)):.4e}")
