# ============================================================
# SOAL 11.14 - Analisis Konvergensi Gauss-Seidel (slope 1 & -1)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("SOAL 11.14 – Gauss-Seidel: Slope 1 dan -1")
print("=" * 50)

# Jika slope kedua persamaan = 1 dan -1:
# x + y = 2   → y = 2 - x  (slope -1)
# x - y = 0   → y = x      (slope +1)
# Solusi: x=1, y=1

A = np.array([[1, 1],
              [1, -1]], dtype=float)
b = np.array([2.0, 0.0])

print("\nSistem: x + y = 2 dan x - y = 0")
print("Solusi eksak: x=1, y=1")

x = np.zeros(2)
history = []
print(f"\n{'Iter':>5} | {'x':>10} | {'y':>10} | {'Error%':>10}")
print("-" * 45)
for it in range(1, 50):
    x_old = x.copy()
    x[0] = (b[0] - x[1]) / A[0,0]
    x[1] = (b[1] - x[0]) / A[1,1]
    err = max(abs((x[i]-x_old[i])/(x[i]+1e-15))*100 for i in range(2))
    history.append((it, x[0], x[1], err))
    print(f"{it:>5} | {x[0]:>10.6f} | {x[1]:>10.6f} | {err:>9.4f}%")
    if err < 0.001:
        print(f"\n✅ Konvergen pada iterasi {it}")
        break

# Plot
iters = [h[0] for h in history]
errs  = [h[3] for h in history]
plt.figure(figsize=(8, 4))
plt.plot(iters, errs, 'b-o', markersize=4)
plt.xlabel("Iterasi"); plt.ylabel("Error (%)"); plt.title("Konvergensi Gauss-Seidel (slope 1 & -1)")
plt.grid(True, alpha=0.3); plt.yscale('log')
plt.tight_layout(); plt.savefig("soal_11_14_plot.png", dpi=120); plt.show()
print("Plot disimpan: soal_11_14_plot.png")
