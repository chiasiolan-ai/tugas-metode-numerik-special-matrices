# ============================================================
# SOAL 11.23 - Jumlah Operasi: Thomas vs Gauss Elimination
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("SOAL 11.23 – Operasi Thomas vs Gauss Elimination")
print("=" * 50)

ns = range(2, 21)

# Thomas algorithm: O(n) — forward + back sweep
# Forward: (n-1) operasi per baris → 3(n-1) ops
# Back sub: 2(n-1) ops
# Total ≈ 5n - 4 flops
def ops_thomas(n):
    return 5*n - 4  # linear

# Gauss elimination tanpa partial pivoting: O(n³)
# Total ≈ (2/3)n³ + O(n²)
def ops_gauss(n):
    return int(2/3 * n**3 + n**2)

print(f"\n{'n':>4} | {'Thomas':>10} | {'Gauss':>12} | {'Rasio':>10}")
print("-" * 45)
for n in ns:
    t = ops_thomas(n)
    g = ops_gauss(n)
    print(f"{n:>4} | {t:>10} | {g:>12} | {g/t:>10.1f}x lebih banyak")

# Plot
t_ops = [ops_thomas(n) for n in ns]
g_ops = [ops_gauss(n) for n in ns]

plt.figure(figsize=(9, 5))
plt.plot(list(ns), t_ops, 'g-o', label='Thomas Algorithm O(n)', linewidth=2, markersize=6)
plt.plot(list(ns), g_ops, 'r-s', label='Gauss Elimination O(n³)', linewidth=2, markersize=6)
plt.xlabel('Ukuran Matriks n')
plt.ylabel('Jumlah Operasi')
plt.title('Perbandingan Operasi: Thomas vs Gauss Elimination')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("soal_11_23_plot.png", dpi=120)
plt.show()
print("\nPlot disimpan: soal_11_23_plot.png")
