# ============================================================
# SOAL 11.17 - Sistem Persamaan Nonlinear (Newton-Raphson)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

print("=" * 50)
print("SOAL 11.17 – Persamaan Nonlinear")
print("=" * 50)
print("f(x,y) = 4 - y - 2x²")
print("g(x,y) = 8 - y² - 4x")

def equations(vars):
    x, y = vars
    f = 4 - y - 2*x**2
    g = 8 - y**2 - 4*x
    return [f, g]

# Cari dua pasang solusi dengan initial guess berbeda
solutions = []
guesses = [(1,1), (-1,1), (1,-1), (-1,-1), (2,0), (-2,0)]
for g in guesses:
    sol = fsolve(equations, g, full_output=True)
    if sol[2] == 1:  # converged
        xy = tuple(np.round(sol[0], 6))
        # Cek residual
        res = np.max(np.abs(equations(sol[0])))
        if res < 1e-8 and xy not in solutions:
            solutions.append(xy)

print(f"\nSolusi yang ditemukan:")
for i, (x,y) in enumerate(solutions):
    print(f"   Pasang {i+1}: x = {x:.6f}, y = {y:.6f}")
    print(f"            f(x,y) = {4-y-2*x**2:.2e}, g(x,y) = {8-y**2-4*x:.2e}")

# (b) Peta initial guess
print("\n(b) Peta initial guess (x,y dari -6 ke 6):")
res_grid = np.zeros((13,13), dtype=int)
xs = ys = range(-6, 7)
for i,xi in enumerate(xs):
    for j,yj in enumerate(ys):
        try:
            sol = fsolve(equations, [xi,yj], full_output=True)
            if sol[2] == 1:
                xy = tuple(np.round(sol[0],3))
                # Tentukan pasang mana
                for k,(sx,sy) in enumerate(solutions):
                    if abs(xy[0]-sx)<0.1 and abs(xy[1]-sy)<0.1:
                        res_grid[i,j] = k+1; break
        except:
            pass

plt.figure(figsize=(7,6))
plt.imshow(res_grid, extent=[-6,6,-6,6], origin='lower', cmap='coolwarm', alpha=0.7)
plt.colorbar(label='Pasang Solusi')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Peta Initial Guess → Solusi')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("soal_11_17_peta.png", dpi=120); plt.show()
print("Plot disimpan: soal_11_17_peta.png")
