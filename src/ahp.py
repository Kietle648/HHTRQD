import numpy as np

# RI (Random Index) theo Saaty cho n=1..10
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}


def ahp_weights(pairwise: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Tính trọng số AHP theo đúng cách trình bày trong file PDF mẫu:
    1) Chuẩn hóa ma trận theo cột
    2) Lấy trung bình theo hàng để ra trọng số
    3) Tính consistency vector, lambda_max, CI, CR

    Trả về:
    - weights (row-average của ma trận chuẩn hóa)
    - info: lambda_max, CI, CR
    """
    A = np.array(pairwise, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Pairwise matrix phải là ma trận vuông.")

    n = int(A.shape[0])

    # 1) Chuẩn hóa theo cột
    col_sums = A.sum(axis=0)
    normalized = np.divide(A, col_sums, out=np.zeros_like(A), where=col_sums != 0)

    # 2) Trọng số = trung bình theo hàng (đúng theo file PDF)
    w = normalized.mean(axis=1)
    if float(w.sum()) == 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = w / w.sum()

    # 3) Consistency
    weighted_sum = A @ w
    consistency_vector = np.divide(weighted_sum, w, out=np.zeros_like(weighted_sum), where=w != 0)
    lambda_max = float(consistency_vector.mean()) if n > 0 else 0.0
    CI = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = (CI / RI) if RI > 1e-12 else 0.0

    info = {"lambda_max": lambda_max, "CI": float(CI), "CR": float(CR)}
    return w.astype(float), info
