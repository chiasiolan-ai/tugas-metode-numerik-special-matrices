# ============================================================
# SOAL 11.9 - Gauss-Seidel: Sistem Reaktor Kimia
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.9 – Gauss-Seidel: Konsentrasi Reaktor")
print("=" * 50)

# 15c1 - 3c2 - c3   = 3800
# -3c1 + 18c2 - 6c3 = 1200
# -4c1 - c2 + 12c3  = 2350
A = np.array([[15, -3, -1],
              [-3, 18, -6],
              [-4, -1, 12]], dtype=float)
b = np.array([3800.0, 1200.0, 2350.0])

# Cek diagonal dominan
print("\nCek diagonal dominan:")
for i in range(3):
    diag = abs(A[i,i])
    off  = sum(abs(A[i,j]) for j in range(3) if j!=i)
    print(f"   Baris {i+1}: |{diag}| {'>' if diag>off else '<='} {off} → {'✅' if diag>off else '❌'}")

x = np.zeros(3)
es = 5.0
print(f"\n{'Iter':>5} | {'c1':>10} | {'c2':>10} | {'c3':>10} | {'Error%':>8}")
print("-" * 55)
for it in range(1, 200):
    x_old = x.copy()
    x[0] = (b[0] + 3*x[1] + x[2]) / 15
    x[1] = (b[1] + 3*x[0] + 6*x[2]) / 18
    x[2] = (b[2] + 4*x[0] + x[1]) / 12
    err = max(abs((x[i]-x_old[i])/(x[i]+1e-15))*100 for i in range(3))
    print(f"{it:>5} | {x[0]:>10.4f} | {x[1]:>10.4f} | {x[2]:>10.4f} | {err:>7.3f}%")
    if err < es:
        print(f"\n✅ Konvergen pada iterasi {it}")
        break

print(f"\nSolusi: c1={x[0]:.4f}, c2={x[1]:.4f}, c3={x[2]:.4f} g/m³")
print("Verifikasi:", np.round(A@x, 2), "≈", b)
