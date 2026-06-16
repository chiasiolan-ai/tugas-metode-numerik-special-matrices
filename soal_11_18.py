# ============================================================
# SOAL 11.18 - Perencanaan Produksi (Sistem Linear)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.18 – Perencanaan Produksi Elektronik")
print("=" * 50)
print("""
Komponen  | Tembaga | Seng | Kaca
--------------------------------------
Transistor|    4    |   1  |   2
Resistor  |    3    |   3  |   1
Chip      |    2    |   1  |   3
--------------------------------------
Tersedia  |   960   |  510 |  610
""")

# 4T + 3R + 2C = 960  (tembaga)
#  T + 3R +  C = 510  (seng)
# 2T +  R + 3C = 610  (kaca)
A = np.array([[4, 3, 2],
              [1, 3, 1],
              [2, 1, 3]], dtype=float)
b = np.array([960.0, 510.0, 610.0])

x = np.linalg.solve(A, b)
print(f"Solusi:")
print(f"   Transistor (T) = {x[0]:.2f} unit")
print(f"   Resistor   (R) = {x[1]:.2f} unit")
print(f"   Chip       (C) = {x[2]:.2f} unit")

print("\nVerifikasi penggunaan material:")
labels = ["Tembaga", "Seng", "Kaca"]
for i in range(3):
    print(f"   {labels[i]:8}: {A[i]@x:.1f} / {b[i]:.1f}")
