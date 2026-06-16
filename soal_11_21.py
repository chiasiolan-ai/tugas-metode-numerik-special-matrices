# ============================================================
# SOAL 11.21 - Augmented Matrix [A | I]
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.21 – Augmented Matrix [A | I]")
print("=" * 50)

# Contoh matriks A (bisa diganti ukuran apapun)
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)

n = A.shape[0]
I = np.eye(n)

# Satu baris Python (ekuivalen MATLAB: Aug = [A, eye(size(A))])
Aug = np.hstack([A, I])

print(f"\nMatriks A ({n}x{n}):")
print(A)

print(f"\nIdentitas I ({n}x{n}):")
print(I)

print(f"\nAugmented Matrix [A | I] ({n}x{2*n}):")
print(Aug)

print("\nPerintah satu baris Python:")
print("   Aug = np.hstack([A, np.eye(A.shape[0])])")

# Verifikasi: Aug[:,n:] seharusnya = I
print("\nBagian kanan Aug (harus = I):")
print(Aug[:, n:])
