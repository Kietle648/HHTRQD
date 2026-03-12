import pandas as pd
from .features import minmax_normalize

def build_scoring_table(criteria: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns expected:
    Amount_total, Profit_total, Quantity_total, Stability, Pred_Amount_1m (optional), Pred_Amount_3m_avg (optional)
    """
    df = criteria.copy()

    # Normalize từng tiêu chí (càng lớn càng tốt)
    for col in ["Amount_total", "Profit_total", "Quantity_total", "Stability",
                "Pred_Amount_1m", "Pred_Amount_3m_avg"]:
        if col in df.columns:
            df[col + "_norm"] = minmax_normalize(df[col].fillna(0.0))

    return df

def score_and_rank(scoring_table: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    weights: dict mapping criterion -> weight
    ví dụ:
      {"Amount_total":0.3, "Profit_total":0.25, "Quantity_total":0.2, "Stability":0.15, "Pred_Amount_1m":0.1}
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