import numpy as np
import pandas as pd

def make_monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output: mỗi dòng là (Sub-Category, YearMonth) với tổng Amount/Profit/Quantity theo tháng
    """
    monthly = (
        df.groupby(["Sub-Category", "YearMonth"], as_index=False)
          .agg({"Amount": "sum", "Profit": "sum", "Quantity": "sum"})
          .sort_values(["Sub-Category", "YearMonth"])
    )
    return monthly

def compute_stability(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Stability dùng CV = std/mean theo Amount theo tháng.
    StabilityScore = 1 / (CV + epsilon) để "càng lớn càng tốt"
    """
    eps = 1e-9

    g = monthly.groupby("Sub-Category")["Amount"]
    mean = g.mean()
    std = g.std(ddof=0)  # ddof=0 ổn định hơn khi ít tháng
    cv = std / (mean + eps)

    stability_score = 1.0 / (cv + eps)

    out = pd.DataFrame({
        "Sub-Category": mean.index,
        "Amount_mean": mean.values,
        "Amount_std": std.values,
        "Amount_cv": cv.values,
        "Stability": stability_score.values,
    })
    return out

def compute_criteria_table(df: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Criteria:
    - Amount_total
    - Profit_total
    - Quantity_total
    - Stability (từ monthly)
    """
    base = (
        df.groupby("Sub-Category", as_index=False)
          .agg({"Amount": "sum", "Profit": "sum", "Quantity": "sum"})
          .rename(columns={
              "Amount": "Amount_total",
              "Profit": "Profit_total",
              "Quantity": "Quantity_total"
          })
    )

    stability = compute_stability(monthly)[["Sub-Category", "Stability"]]
    crit = base.merge(stability, on="Sub-Category", how="left")
    return crit

def minmax_normalize(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if abs(mx - mn) < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)