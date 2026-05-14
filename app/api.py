from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import core modules của bạn trong src/
from src.data_io import load_csv
from src.preprocess import standardize_columns, ensure_yearmonth, validate_required_columns
from src.features import make_monthly_table, compute_criteria_table
from src.ml import train_next_month_model, predict_3_months_ahead
from src.ahp import ahp_weights
from src.ranker import score_and_rank_ahp
from pathlib import Path




APP_TITLE = "DSS AHP + ML Backend"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Sales Dataset.csv"

app = FastAPI(title=APP_TITLE)

# ---- Cache dữ liệu đã load để API nhanh hơn
RAW_DF: Optional[pd.DataFrame] = None


# =========================
# Pydantic Schemas
# =========================
class RankRequest(BaseModel):
    # Filter
    start_month: Optional[str] = Field(default=None, description="YYYY-MM (vd: 2024-01)")
    end_month: Optional[str] = Field(default=None, description="YYYY-MM (vd: 2024-12)")
    subcategories: Optional[List[str]] = Field(default=None, description="Danh sách Sub-Category cần lọc (optional)")

    # AHP pairwise matrix 4x4 theo thứ tự:
    # [Amount_total, Profit_total, Quantity_total, Stability]
    pairwise_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="Ma trận 4x4 AHP. Nếu null sẽ dùng ma trận mặc định."
    )

    # Output
    top_n: int = Field(default=10, ge=1, le=100)
    horizon: str = Field(default="1m", description="1m hoặc 3m (chỉ để hiển thị ML prediction)")


class RankItem(BaseModel):
    Rank: int
    SubCategory: str
    Score: float
    Amount_total: float
    Profit_total: float
    Quantity_total: float
    Stability: float
    Pred_Amount_1m: Optional[float] = None
    Pred_Amount_3m_avg: Optional[float] = None
    # breakdown điểm theo tiêu chí = trọng số tiêu chí * trọng số phương án theo tiêu chí
    contrib: Dict[str, float]
    # trọng số phương án theo từng tiêu chí dùng để tính Score theo công thức AHP PDF
    local_weights: Dict[str, float]


class RankResponse(BaseModel):
    meta: Dict[str, Any]
    ahp: Dict[str, Any]
    ml: Dict[str, Any]
    results: List[RankItem]


# =========================
# Helpers
# =========================
def _load_data_once() -> pd.DataFrame:
    global RAW_DF
    if RAW_DF is not None:
        return RAW_DF

    df = load_csv(DATA_PATH)
    df = standardize_columns(df)
    df = ensure_yearmonth(df)

    # validate required columns (Sub-Category, Amount, Profit, Quantity, YearMonth)
    validate_required_columns(df)

    RAW_DF = df
    return RAW_DF


def _filter_df(df: pd.DataFrame, start_month: Optional[str], end_month: Optional[str],
               subcategories: Optional[List[str]]) -> pd.DataFrame:
    out = df.copy()

    # Filter by month range
    if start_month:
        out = out[out["YearMonth"] >= start_month]
    if end_month:
        out = out[out["YearMonth"] <= end_month]

    # Filter by subcategories
    if subcategories:
        out = out[out["Sub-Category"].isin(subcategories)]

    return out


def _default_pairwise_4x4() -> np.ndarray:
    # Ma trận mặc định giống Excel của bạn:
    # Amount vs Profit = 3
    # Amount vs Quantity = 2
    # Amount vs Stability = 5
    # Profit vs Quantity = 1/2
    # Profit vs Stability = 3
    # Quantity vs Stability = 4
    return np.array([
        [1,   3,   2,   5],
        [1/3, 1,   1/2, 3],
        [1/2, 2,   1,   4],
        [1/5, 1/3, 1/4, 1],
    ], dtype=float)


def _validate_pairwise(m: List[List[float]]) -> np.ndarray:
    A = np.array(m, dtype=float)
    if A.shape != (4, 4):
        raise ValueError("pairwise_matrix phải là ma trận 4x4.")
    # không ép user phải đúng nghịch đảo 1/x, nhưng khuyên nên đúng
    if not np.allclose(np.diag(A), 1.0):
        raise ValueError("Đường chéo chính của pairwise_matrix phải = 1.")
    return A


def _to_items(ranked_df: pd.DataFrame,
              weights: Dict[str, float],
              horizon: str) -> List[RankItem]:
    pred1 = "Pred_Amount_1m"
    pred3 = "Pred_Amount_3m_avg"

    items: List[RankItem] = []
    for _, row in ranked_df.iterrows():
        contrib: Dict[str, float] = {}
        local_weights: Dict[str, float] = {}
        for crit, w in weights.items():
            local_col = f"{crit}_local_weight"
            contrib_col = f"{crit}_contrib"
            local_weights[crit] = float(row[local_col]) if local_col in row else 0.0
            contrib[crit] = float(row[contrib_col]) if contrib_col in row else float(w) * local_weights[crit]

        items.append(RankItem(
            Rank=int(row["Rank"]),
            SubCategory=str(row["Sub-Category"]),
            Score=float(row["Score"]),
            Amount_total=float(row["Amount_total"]),
            Profit_total=float(row["Profit_total"]),
            Quantity_total=float(row["Quantity_total"]),
            Stability=float(row["Stability"]),
            Pred_Amount_1m=(float(row[pred1]) if pred1 in row and not pd.isna(row[pred1]) else None),
            Pred_Amount_3m_avg=(float(row[pred3]) if pred3 in row and not pd.isna(row[pred3]) else None),
            contrib=contrib,
            local_weights=local_weights
        ))
    return items


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "service": APP_TITLE}


@app.get("/meta")
def meta():
    df = _load_data_once()
    subcats = sorted(df["Sub-Category"].dropna().unique().tolist())
    months = sorted(df["YearMonth"].dropna().unique().tolist())

    return {
        "subcategories": subcats,
        "months": {
            "all": months,
            "min": months[0] if months else None,
            "max": months[-1] if months else None
        },
        "criteria_order": ["Amount_total", "Profit_total", "Quantity_total", "Stability"],
        "pairwise_default": _default_pairwise_4x4().tolist(),
        "horizons": ["1m", "3m"]
    }


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    df = _load_data_once()

    # 1) filter data
    df_f = _filter_df(df, req.start_month, req.end_month, req.subcategories)
    if df_f.empty:
        raise HTTPException(status_code=400, detail="Filter ra dữ liệu rỗng. Hãy chọn lại start/end/subcategories.")

    # 2) build monthly + criteria
    monthly = make_monthly_table(df_f)
    crit = compute_criteria_table(df_f, monthly)

    # 3) ML train + predict (tham khảo)
    artifacts, ml_metrics = train_next_month_model(monthly)
    preds = predict_3_months_ahead(monthly, artifacts)
    crit = crit.merge(
        preds[["Sub-Category", "Pred_Amount_1m", "Pred_Amount_3m_avg"]],
        on="Sub-Category",
        how="left"
    )

    # 4) AHP weights (4 tiêu chí)
    criteria_order = ["Amount_total", "Profit_total", "Quantity_total", "Stability"]

    try:
        if req.pairwise_matrix is None:
            A = _default_pairwise_4x4()
        else:
            A = _validate_pairwise(req.pairwise_matrix)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    w, ahp_info = ahp_weights(A)
    weights = {criteria_order[i]: float(w[i]) for i in range(4)}

    if float(ahp_info.get("CR", 0.0)) >= 0.1:
        raise HTTPException(
            status_code=400,
            detail="Ma trận so sánh cặp tiêu chí chưa nhất quán (CR >= 0.1). Hãy điều chỉnh lại trước khi xếp hạng."
        )

    # 5) scoring + ranking theo công thức AHP PDF:
    # Score(i) = sum_k [ w_k * p_(i,k) ]
    ranked_all, _ = score_and_rank_ahp(
        criteria_table=crit,
        weights=weights,
        criteria_order=criteria_order,
    )
    ranked = ranked_all.head(req.top_n)

    # 6) response
    items = _to_items(ranked, weights, req.horizon)

    return RankResponse(
        meta={
            "start_month": req.start_month,
            "end_month": req.end_month,
            "top_n": req.top_n,
            "horizon": req.horizon,
            "n_subcategories_returned": len(items)
        },
        ahp={
            "criteria_order": criteria_order,
            "pairwise_matrix_used": A.tolist(),
            "weights": weights,
            "consistency": ahp_info
        },
        ml={
            "note": "ML chỉ dùng tham khảo (không đưa vào AHP score). Độ sai lệch trung bình tuyệt đối:",
            "metrics": ml_metrics
        },
        results=items
    )