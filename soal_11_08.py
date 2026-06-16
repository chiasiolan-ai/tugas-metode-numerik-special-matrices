# ============================================================
# SOAL 11.8 - Gauss-Seidel dengan Overrelaxation (λ=1.2)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.8 – Gauss-Seidel + Overrelaxation (λ=1.2)")
print("=" * 50)

# Sistem dari Soal 11.1
A = np.array([[0.8, -0.4, 0.0],
              [-0.4, 0.8, -0.4],
              [0.0, -0.4, 0.8]], dtype=float)
b = np.array([41.0, 25.0, 105.0])

def gauss_seidel(A, b, lam=1.0, es=0.05, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    print(f"\n{'Iter':>5} | {'x1':>12} | {'x2':>12} | {'x3':>12} | {'Max Error':>12}")
    print("-" * 65)
    for it in range(1, max_iter+1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i,j]*x[j] for j in range(n) if j!=i)
            x_new = (b[i] - sigma) / A[i,i]
            x[i] = lam*x_new + (1-lam)*x_old[i]
        # Hitung error relatif
        err = max(abs((x[i]-x_old[i])/(x[i]+1e-15)) for i in range(n)) * 100
        print(f"{it:>5} | {x[0]:>12.6f} | {x[1]:>12.6f} | {x[2]:>12.6f} | {err:>11.4f}%")
        if err < es*100:
            print(f"\n✅ Konvergen pada iterasi {it} (error < {es*100}%)")
            break
    return x

x = gauss_seidel(A, b, lam=1.2, es=0.05)
print(f"\nSolusi akhir: x1={x[0]:.4f}, x2={x[1]:.4f}, x3={x[2]:.4f}")
