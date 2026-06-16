# ============================================================
# SOAL 11.19 - Hilbert Matrix 10x10: Condition Number
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.19 – Hilbert Matrix 10x10")
print("=" * 50)

n = 10
# Buat Hilbert matrix: H[i,j] = 1/(i+j+1)
H = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])

print(f"\nHilbert Matrix {n}x{n} (3 baris pertama):")
print(np.round(H[:3,:3], 6))

# Condition number spektral
cond = np.linalg.cond(H)
print(f"\nCondition Number (spektral): {cond:.4e}")
digits_lost = np.log10(cond)
print(f"Perkiraan digit presisi hilang: {digits_lost:.1f} digit")

# Solve: b[i] = sum baris ke-i (solusi x seharusnya semua = 1)
b = H.sum(axis=1)
x = np.linalg.solve(H, b)

print(f"\nSolusi (seharusnya semua = 1):")
for i in range(n):
    print(f"   x{i+1:2d} = {x[i]:>12.6f}  error = {abs(x[i]-1):.2e}")

max_err = np.max(np.abs(x - 1))
print(f"\nMaks error aktual : {max_err:.4e}")
print(f"Error prediksi    : ~1e-{16-int(digits_lost):.0f} (dari condition number)")
