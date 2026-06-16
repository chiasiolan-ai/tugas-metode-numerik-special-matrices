# ============================================================
# SOAL 11.11 - Gauss-Seidel es < 5%
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.11 – Gauss-Seidel (es < 5%)")
print("=" * 50)

# 10x1 + 2x2 - x3  = 27
# -3x1 - 6x2 + 2x3 = -61.5
#  x1 + x2 + 5x3   = -21.5
A = np.array([[10, 2, -1],
              [-3, -6, 2],
              [1, 1, 5]], dtype=float)
b = np.array([27.0, -61.5, -21.5])

print("\nCek diagonal dominan:")
for i in range(3):
    d = abs(A[i,i])
    o = sum(abs(A[i,j]) for j in range(3) if j!=i)
    print(f"   Baris {i+1}: |{d}| {'>' if d>o else '<='} {o} {'✅' if d>o else '❌'}")

x = np.zeros(3)
es = 5.0
print(f"\n{'Iter':>5} | {'x1':>10} | {'x2':>10} | {'x3':>10} | {'Error%':>8}")
print("-" * 55)
for it in range(1, 200):
    x_old = x.copy()
    x[0] = (b[0] - 2*x[1] + x[2]) / 10
    x[1] = (b[1] + 3*x[0] - 2*x[2]) / (-6)
    x[2] = (b[2] - x[0] - x[1]) / 5
    err = max(abs((x[i]-x_old[i])/(x[i]+1e-15))*100 for i in range(3))
    print(f"{it:>5} | {x[0]:>10.4f} | {x[1]:>10.4f} | {x[2]:>10.4f} | {err:>7.3f}%")
    if err < es:
        print(f"\n✅ Konvergen pada iterasi {it}")
        break

print(f"\nSolusi: x1={x[0]:.4f}, x2={x[1]:.4f}, x3={x[2]:.4f}")
print("Verifikasi:", np.round(A@x,2), "≈", b)
