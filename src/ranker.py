from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .ahp import ahp_weights
from .features import minmax_normalize


def build_scoring_table(criteria: pd.DataFrame) -> pd.DataFrame:
    """
    Giữ nguyên hàm cũ để tránh làm vỡ các luồng đang dùng normalize + weighted sum.
    Không dùng trong /rank sau khi chuyển sang full AHP cho phương án.
    """
    df = criteria.copy()

    for col in [
        "Amount_total",
        "Profit_total",
        "Quantity_total",
        "Stability",
        "Pred_Amount_1m",
        "Pred_Amount_3m_avg",
    ]:
        if col in df.columns:
            df[col + "_norm"] = minmax_normalize(df[col].fillna(0.0))

    return df


def _positive_scores(values: Iterable[float]) -> np.ndarray:
    """
    Quy đổi dữ liệu thô về dãy số dương để dựng ma trận so sánh cặp.
    Với dataset hiện tại các tiêu chí đều dương, nên giá trị thường giữ nguyên.
    Chỉ shift khi có 0 hoặc số âm để tránh chia cho 0.
    """
    arr = pd.to_numeric(pd.Series(list(values), dtype="float64"), errors="coerce").fillna(0.0).to_numpy()
    if arr.size == 0:
        return arr.astype(float)

    mn = float(np.min(arr))
    if mn <= 0:
        arr = arr - mn + 1.0

    return arr.astype(float)


def build_pairwise_alternative_matrix(values: Iterable[float]) -> np.ndarray:
    """
    Hệ thống tự tạo ma trận phương án theo từng tiêu chí.
    Công thức cốt lõi vẫn là ma trận so sánh cặp AHP giữa các phương án,
    nhưng user không cần nhập tay.
    """
    scores = _positive_scores(values)
    n = int(scores.shape[0])
    pairwise = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            den = float(scores[j])
            pairwise[i, j] = 1.0 if abs(den) < 1e-12 else float(scores[i]) / den

    return pairwise


def build_alternative_priority_table(
    criteria: pd.DataFrame,
    criteria_order: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Với mỗi tiêu chí:
    1) Tự sinh ma trận phương án
    2) Tính trọng số cục bộ của các phương án theo tiêu chí đó
    3) Ghi lại vào cột <criterion>_local_weight
    """
    df = criteria.copy()
    details: Dict[str, Dict[str, object]] = {}

    for crit in criteria_order:
        if crit not in df.columns:
            raise ValueError(f"Thiếu cột tiêu chí: {crit}")

        raw_values = pd.to_numeric(df[crit], errors="coerce").fillna(0.0)
        pairwise = build_pairwise_alternative_matrix(raw_values)
        local_weights, info = ahp_weights(pairwise)

        df[f"{crit}_local_weight"] = local_weights

        details[crit] = {
            "source_values": raw_values.astype(float).tolist(),
            "pairwise_matrix": pairwise.round(6).tolist(),
            "local_weights": local_weights.round(6).tolist(),
            "consistency": info,
        }

    return df, details


def score_and_rank_ahp(
    criteria_table: pd.DataFrame,
    weights: Dict[str, float],
    criteria_order: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Công thức đúng theo file PDF:
    Score(i) = sum_k [ w_k * p_(i,k) ]
    trong đó p_(i,k) là trọng số phương án i theo tiêu chí k.
    """
    if criteria_order is None:
        criteria_order = list(weights.keys())

    df, details = build_alternative_priority_table(criteria_table, criteria_order)

    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for crit in criteria_order:
        local_col = f"{crit}_local_weight"
        contrib_col = f"{crit}_contrib"
        df[contrib_col] = float(weights[crit]) * df[local_col]
        score = score + df[contrib_col]

    df["Score"] = score
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df, details


def score_and_rank(scoring_table: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Giữ nguyên hàm cũ để tránh ảnh hưởng các phần đang ổn.
    """
    df = scoring_table.copy()
    score = 0.0

    for crit, w in weights.items():
        norm_col = crit + "_norm"
        if norm_col not in df.columns:
            raise ValueError(f"Thiếu cột đã normalize cho tiêu chí: {crit} (cần {norm_col})")
        score = score + float(w) * df[norm_col]

    df["Score"] = score
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df
