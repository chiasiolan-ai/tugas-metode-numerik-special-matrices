# ============================================================
# SOAL 11.26 - Program Gauss-Seidel (User-Friendly)
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

def gauss_seidel(A, b, lam=1.0, es=5.0, max_iter=100, verbose=True):
    """
    Metode Gauss-Seidel dengan opsi overrelaxation.

    Parameter:
    ----------
    A       : matriks koefisien
    b       : vector RHS
    lam     : faktor relaxasi (default=1.0, tanpa relaxasi)
    es      : toleransi error (%) — default 5%
    max_iter: iterasi maksimum
    verbose : tampilkan tabel iterasi

    Return:
    -------
    x    : solusi
    iters: jumlah iterasi sampai konvergen
    """
    n = len(b)
    x = np.zeros(n)

    # Cek diagonal dominan
    print("\n📌 Cek Diagonal Dominan:")
    all_dom = True
    for i in range(n):
        d = abs(A[i,i])
        o = sum(abs(A[i,j]) for j in range(n) if j!=i)
        dom = d > o
        if not dom: all_dom = False
        print(f"   Baris {i+1}: |{d:.4f}| {'>' if dom else '<='} {o:.4f} {'✅' if dom else '⚠️'}")
    if not all_dom:
        print("   ⚠️  Tidak sepenuhnya diagonal dominan — konvergensi tidak dijamin!")

    if verbose:
        header = f"{'Iter':>5} | " + " | ".join(f"{'x'+str(i+1):>10}" for i in range(n)) + f" | {'MaxErr(%)':>10}"
        print("\n" + header)
        print("-" * len(header))

    for it in range(1, max_iter+1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i,j]*x[j] for j in range(n) if j!=i)
            x_new = (b[i] - sigma) / A[i,i]
            x[i] = lam*x_new + (1-lam)*x_old[i]

        max_err = max(abs((x[i]-x_old[i])/(x[i]+1e-15))*100 for i in range(n))

        if verbose:
            row = f"{it:>5} | " + " | ".join(f"{x[i]:>10.5f}" for i in range(n)) + f" | {max_err:>9.4f}%"
            print(row)

        if max_err < es:
            if verbose:
                print(f"\n✅ Konvergen pada iterasi {it} (error {max_err:.4f}% < {es}%)")
            return x, it

    print(f"\n⚠️  Tidak konvergen dalam {max_iter} iterasi")
    return x, max_iter


# ── Test dengan Example 11.3 ────────────────────────────────
print("=" * 60)
print("SOAL 11.26 – Program Gauss-Seidel")
print("=" * 60)
print("\nTest: Sistem dari Example 11.3")
print("   10x1 + 2x2 - x3  = 27")
print("   -3x1 - 6x2 + 2x3 = -61.5")
print("    x1 +  x2 + 5x3  = -21.5")

A = np.array([[10, 2, -1],
              [-3, -6, 2],
              [1, 1, 5]], dtype=float)
b = np.array([27.0, -61.5, -21.5])

x, it = gauss_seidel(A, b, lam=1.0, es=5.0)

print(f"\nSolusi akhir:")
for i, xi in enumerate(x):
    print(f"   x{i+1} = {xi:.6f}")
print(f"\nVerifikasi A*x ≈ b:")
print(f"   {np.round(A@x,3)} ≈ {b}")
