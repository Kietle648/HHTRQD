import pandas as pd
from .data_io import load_csv
from .preprocess import standardize_columns, ensure_yearmonth, validate_required_columns
from .features import make_monthly_table, compute_criteria_table
from .ml import train_next_month_model, predict_3_months_ahead, save_model
from .ranker import build_scoring_table, score_and_rank

def run_pipeline(
    csv_path: str,
    use_ml: bool = True,
    top_n: int = 10,
    weights: dict | None = None,
    save_model_path: str | None = None,
) -> dict:
    """
    weights mặc định (nếu bạn chưa dùng AHP UI):
    """
    if weights is None:
        weights = {
            "Amount_total": 0.30,
            "Profit_total": 0.25,
            "Quantity_total": 0.20,
            "Stability": 0.15,
            "Pred_Amount_1m": 0.10,  # nếu use_ml=False thì bỏ sau
        }

    # 1) Load + preprocess
    df = load_csv(csv_path)
    df = standardize_columns(df)
    df = ensure_yearmonth(df)

    # validate required columns
    validate_required_columns(df)

    # 2) Monthly table
    monthly = make_monthly_table(df)

    # 3) Criteria AHP
    crit = compute_criteria_table(df, monthly)

    ml_metrics = None
    if use_ml:
        # 4) ML train + predict
        artifacts, ml_metrics = train_next_month_model(monthly)

        if save_model_path:
            save_model(artifacts, save_model_path)

        preds = predict_3_months_ahead(monthly, artifacts)
        crit = crit.merge(preds[["Sub-Category", "Pred_Amount_1m", "Pred_Amount_3m_avg"]], on="Sub-Category", how="left")
    else:
        # nếu không dùng ML, remove trọng số ML nếu có
        if "Pred_Amount_1m" in weights:
            weights = {k: v for k, v in weights.items() if k != "Pred_Amount_1m"}

    # 5) Scoring + ranking
    scoring = build_scoring_table(crit)
    ranked = score_and_rank(scoring, weights)

    return {
        "monthly": monthly,
        "criteria": crit,
        "ranked": ranked.head(top_n),
        "ml_metrics": ml_metrics,
        "weights": weights
    }