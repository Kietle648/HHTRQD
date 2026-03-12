import numpy as np

# RI (Random Index) theo Saaty cho n=1..10
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

def ahp_weights(pairwise: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Trả về:
    - weights (eigenvector normalized)
    - info: lambda_max, CI, CR
    """
    A = np.array(pairwise, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Pairwise matrix phải là ma trận vuông.")

    # Eigenvector principal
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    lambda_max = float(eigvals.real[idx])
    w = eigvecs[:, idx].real
    w = w / w.sum()

    # Consistency
    CI = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    RI = RI_TABLE.get(n, 1.49)  # fallback
    CR = (CI / RI) if RI > 1e-12 else 0.0

    info = {"lambda_max": lambda_max, "CI": float(CI), "CR": float(CR)}
    return w.astype(float), info