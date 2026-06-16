# ============================================================
# SOAL 11.15 - Identifikasi Konvergensi Gauss-Seidel
# Nama : GRACIA MARSELINA / NIM : F5512510024
# ============================================================
import numpy as np

print("=" * 50)
print("SOAL 11.15 – Konvergensi 3 Set Persamaan Linear")
print("=" * 50)

sets = {
    "Set One": {
        "A": np.array([[8,3,1],[-6,0,7],[2,4,-1]], dtype=float),
        "b": np.array([12.0,1.0,5.0])
    },
    "Set Two": {
        "A": np.array([[1,1,5],[1,4,-1],[3,1,-1]], dtype=float),
        "b": np.array([7.0,4.0,4.0])
    },
    "Set Three": {
        "A": np.array([[2,3,5],[-2,4,-5],[0,2,-1]], dtype=float),
        "b": np.array([7.0,-3.0,1.0])
    }
}

for name, s in sets.items():
    A, b = s["A"], s["b"]
    print(f"\n{'─'*50}")
    print(f"{name}")
    # Cek diagonal dominan
    converge = True
    for i in range(A.shape[0]):
        d = abs(A[i,i]); o = sum(abs(A[i,j]) for j in range(A.shape[1]) if j!=i)
        dom = d > o
        if not dom: converge = False
        print(f"   Baris {i+1}: |{d}| vs {o} {'✅ Dominan' if dom else '❌ Tidak Dominan'}")
    print(f"   → Diagonal Dominan: {'Ya → Konvergen' if converge else 'Tidak → Mungkin Tidak Konvergen'}")

    # Coba Gauss-Seidel 30 iterasi
    x = np.zeros(A.shape[1])
    prev_err = None
    diverging = False
    for it in range(1, 31):
        x_old = x.copy()
        for i in range(A.shape[0]):
            sigma = sum(A[i,j]*x[j] for j in range(A.shape[1]) if j!=i)
            if abs(A[i,i]) > 1e-15:
                x[i] = (b[i]-sigma)/A[i,i]
        err = np.linalg.norm(x-x_old)
        if prev_err is not None and err > prev_err*2:
            diverging = True
        prev_err = err
        if err < 1e-6: break

    if diverging:
        print(f"   ⚠️  DIVERGEN setelah {it} iterasi")
    else:
        sol = np.linalg.solve(A, b)
        print(f"   Solusi: x={sol[0]:.4f}, y={sol[1]:.4f}, z={sol[2]:.4f}")
