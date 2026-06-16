# ============================================================
# SOAL 11.24 - Program Thomas Algorithm (User-Friendly)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

def thomas_algorithm(a, b, c, d, verbose=True):
    """
    Algoritma Thomas untuk sistem tridiagonal.
    
    Parameter:
    ----------
    a : array subdiagonal    (a[0] tidak dipakai)
    b : array diagonal utama
    c : array superdiagonal  (c[n-1] tidak dipakai)
    d : array sisi kanan (RHS)
    
    Return:
    -------
    x : solusi
    """
    n = len(d)
    b_ = b.copy().astype(float)
    c_ = c.copy().astype(float)
    d_ = d.copy().astype(float)

    if verbose:
        print("\n--- Forward Elimination ---")

    # Forward sweep
    for i in range(1, n):
        factor = a[i] / b_[i-1]
        b_[i] -= factor * c_[i-1]
        d_[i] -= factor * d_[i-1]
        if verbose:
            print(f"   i={i}: faktor={factor:.4f}, b[{i}]={b_[i]:.4f}, d[{i}]={d_[i]:.4f}")

    if verbose:
        print("\n--- Back Substitution ---")

    # Back substitution
    x = np.zeros(n)
    x[-1] = d_[-1] / b_[-1]
    if verbose:
        print(f"   x[{n-1}] = {x[-1]:.6f}")
    for i in range(n-2, -1, -1):
        x[i] = (d_[i] - c_[i] * x[i+1]) / b_[i]
        if verbose:
            print(f"   x[{i}] = {x[i]:.6f}")

    return x


# ── Test dengan data dari Example 11.1 ──────────────────────
print("=" * 55)
print("SOAL 11.24 – Program Thomas Algorithm")
print("=" * 55)
print("\nTest: Sistem dari Example 11.1")
print("[0.8  -0.4       ] [x1]   [41 ]")
print("[-0.4  0.8  -0.4 ] [x2] = [25 ]")
print("[      -0.4  0.8 ] [x3]   [105]")

b = np.array([0.8,  0.8,  0.8])
a = np.array([0.0, -0.4, -0.4])
c = np.array([-0.4, -0.4,  0.0])
d = np.array([41.0, 25.0, 105.0])

x = thomas_algorithm(a, b, c, d, verbose=True)

print(f"\nSolusi:")
for i, xi in enumerate(x):
    print(f"   x{i+1} = {xi:.6f}")

# Verifikasi
A_full = np.array([[0.8,-0.4,0],[-0.4,0.8,-0.4],[0,-0.4,0.8]])
print(f"\nVerifikasi A*x = d: {np.round(A_full@x, 4)} ≈ {d}")

# ── Input interaktif ────────────────────────────────────────
print("\n" + "="*55)
print("Mode Input Manual (opsional, tekan Ctrl+C untuk skip)")
try:
    n = int(input("\nMasukkan ukuran sistem n: "))
    print("Masukkan diagonal utama b:")
    b2 = np.array([float(input(f"  b[{i}] = ")) for i in range(n)])
    print("Masukkan subdiagonal a (a[0] = 0):")
    a2 = np.array([float(input(f"  a[{i}] = ")) for i in range(n)])
    print("Masukkan superdiagonal c (c[n-1] = 0):")
    c2 = np.array([float(input(f"  c[{i}] = ")) for i in range(n)])
    print("Masukkan RHS d:")
    d2 = np.array([float(input(f"  d[{i}] = ")) for i in range(n)])
    x2 = thomas_algorithm(a2, b2, c2, d2)
    print("\nSolusi:")
    for i, xi in enumerate(x2):
        print(f"   x{i+1} = {xi:.6f}")
except (KeyboardInterrupt, EOFError):
    print("\n(Mode input manual dilewati)")
