# ============================================================
# SOAL 11.13 - Gauss-Seidel (a) tanpa & (b) dengan relaxasi λ=1.2
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.13 – Gauss-Seidel ± Relaxasi (λ=1.2)")
print("=" * 50)

# Sistem asli:
#  2x1 - 6x2 - x3  = -38
# -3x1 - x2 + 7x3  = -34
# -8x1 + x2 - 2x3  = -20

# Susun ulang agar diagonal dominan:
# Baris 3 → x1: -8x1 + x2 - 2x3 = -20  (|−8| > |1|+|−2| = 3 ✅)
# Baris 1 → x2:  2x1 - 6x2 - x3 = -38  (|−6| > |2|+|−1| = 3 ✅)
# Baris 2 → x3: -3x1 - x2 + 7x3 = -34  (|7| > |−3|+|−1| = 4 ✅)
A = np.array([[-8,  1, -2],
              [ 2, -6, -1],
              [-3, -1,  7]], dtype=float)
b = np.array([-20.0, -38.0, -34.0])

print("\nSetelah penyusunan ulang:")
for i in range(3):
    d = abs(A[i,i]); o = sum(abs(A[i,j]) for j in range(3) if j!=i)
    print(f"   Baris {i+1}: |{d}| vs {o} {'✅' if d>o else '❌'}")

def gs_solve(A, b, lam, label):
    x = np.zeros(3); es = 5.0
    print(f"\n{label}")
    print(f"{'Iter':>5}|{'x1':>10}|{'x2':>10}|{'x3':>10}|{'Error%':>9}")
    print("-"*55)
    for it in range(1, 300):
        x_old = x.copy()
        for i in range(3):
            sigma = sum(A[i,j]*x[j] for j in range(3) if j!=i)
            xnew = (b[i]-sigma)/A[i,i]
            x[i] = lam*xnew + (1-lam)*x_old[i]
        err = max(abs((x[i]-x_old[i])/(x[i]+1e-15))*100 for i in range(3))
        print(f"{it:>5}|{x[0]:>10.4f}|{x[1]:>10.4f}|{x[2]:>10.4f}|{err:>8.3f}%")
        if err < es:
            print(f"✅ Konvergen iterasi {it}"); break
    return x

x1 = gs_solve(A, b, 1.0, "(a) Tanpa relaxasi")
x2 = gs_solve(A, b, 1.2, "(b) Relaxasi λ=1.2")
print(f"\n(a) x1={x1[0]:.4f}, x2={x1[1]:.4f}, x3={x1[2]:.4f}")
print(f"(b) x1={x2[0]:.4f}, x2={x2[1]:.4f}, x3={x2[2]:.4f}")
