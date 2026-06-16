# ============================================================
# SOAL 11.12 - Gauss-Seidel (a) tanpa & (b) dengan relaxasi λ=0.95
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.12 – Gauss-Seidel ± Relaxasi (λ=0.95)")
print("=" * 50)

# Sistem asli (perlu diurutkan ulang agar diagonal dominan):
# -3x1 + x2 + 12x3 = 50  → tukar jadi baris ke-3
#  6x1 - x2 - x3   = 3   → baris ke-1
#  6x1 + 9x2 + x3  = 40  → baris ke-2
A = np.array([[6, -1, -1],
              [6,  9,  1],
              [-3, 1, 12]], dtype=float)
b = np.array([3.0, 40.0, 50.0])

print("\nSetelah penyusunan ulang (agar diagonal dominan):")
print(A)
print("Cek diagonal dominan:")
for i in range(3):
    d = abs(A[i,i])
    o = sum(abs(A[i,j]) for j in range(3) if j!=i)
    print(f"   Baris {i+1}: |{d}| {'>' if d>o else '<='} {o} {'✅' if d>o else '❌'}")

def gs_solve(A, b, lam, label):
    x = np.zeros(3)
    es = 5.0
    print(f"\n{'─'*60}")
    print(f"{label}")
    print(f"{'Iter':>5} | {'x1':>10} | {'x2':>10} | {'x3':>10} | {'Error%':>8}")
    print("-"*55)
    for it in range(1, 300):
        x_old = x.copy()
        for i in range(3):
            sigma = sum(A[i,j]*x[j] for j in range(3) if j!=i)
            xnew = (b[i]-sigma)/A[i,i]
            x[i] = lam*xnew + (1-lam)*x_old[i]
        err = max(abs((x[i]-x_old[i])/(x[i]+1e-15))*100 for i in range(3))
        print(f"{it:>5} | {x[0]:>10.4f} | {x[1]:>10.4f} | {x[2]:>10.4f} | {err:>7.3f}%")
        if err < es:
            print(f"✅ Konvergen iterasi {it}"); break
    return x

x1 = gs_solve(A, b, 1.0,  "(a) Tanpa relaxasi (λ=1.0)")
x2 = gs_solve(A, b, 0.95, "(b) Dengan relaxasi (λ=0.95)")
print(f"\nSolusi (a): x1={x1[0]:.4f}, x2={x1[1]:.4f}, x3={x1[2]:.4f}")
print(f"Solusi (b): x1={x2[0]:.4f}, x2={x2[1]:.4f}, x3={x2[2]:.4f}")
