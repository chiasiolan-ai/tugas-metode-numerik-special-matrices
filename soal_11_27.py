# ============================================================
# SOAL 11.27 - Persamaan Diferensial: Kanal 1D
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("SOAL 11.27 – PDE Kanal 1D: D*c'' - U*c' - k*c = 0")
print("=" * 55)

# Parameter
D  = 2.0    # koefisien difusi
U  = 1.0    # kecepatan aliran
k  = 0.2    # laju peluruhan orde-1
c0 = 80.0   # c(0) = 80
cL = 20.0   # c(10) = 20
dx = 2.0    # step size
L  = 10.0   # panjang kanal

# Titik interior: x = 2, 4, 6, 8  (4 titik)
xs = np.arange(dx, L, dx)
n  = len(xs)
print(f"\nParameter: D={D}, U={U}, k={k}")
print(f"BC: c(0)={c0}, c(10)={cL}, Δx={dx}")
print(f"Titik interior: {xs}")

# Diskretisasi FD:
# D*(c[i-1]-2c[i]+c[i+1])/dx² - U*(c[i+1]-c[i-1])/(2dx) - k*c[i] = 0
# Koefisien:
alpha = D/dx**2 - U/(2*dx)   # c[i-1]
beta  = -2*D/dx**2 - k       # c[i]
gamma = D/dx**2 + U/(2*dx)   # c[i+1]

print(f"\nKoefisien FD: α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}")

A = np.zeros((n, n))
b = np.zeros(n)

for i in range(n):
    A[i, i] = beta
    if i > 0:
        A[i, i-1] = alpha
    else:
        b[i] -= alpha * c0    # BC kiri
    if i < n-1:
        A[i, i+1] = gamma
    else:
        b[i] -= gamma * cL    # BC kanan

print("\nMatriks A:"); print(np.round(A, 4))
print(f"Vector b: {np.round(b, 4)}")

c_interior = np.linalg.solve(A, b)
c_all = np.concatenate([[c0], c_interior, [cL]])
x_all = np.concatenate([[0], xs, [L]])

print(f"\nSolusi Konsentrasi:")
for xi, ci in zip(x_all, c_all):
    print(f"   c({xi:.0f}) = {ci:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_all, c_all, 'b-o', linewidth=2, markersize=8, label='Solusi FD')
plt.xlabel('Jarak x (m)')
plt.ylabel('Konsentrasi c')
plt.title('Distribusi Konsentrasi dalam Kanal 1D\n(D=2, U=1, k=0.2)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("soal_11_27_plot.png", dpi=120)
plt.show()
print("\nPlot disimpan: soal_11_27_plot.png")
