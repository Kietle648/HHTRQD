import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

@dataclass
class MLArtifacts:
    model: RandomForestRegressor
    feature_cols: list

def add_lags(monthly: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    df = monthly.copy()
    for k in range(1, max_lag + 1):
        df[f"lag{k}"] = df.groupby("Sub-Category")["Amount"].shift(k)
    return df

def train_next_month_model(monthly: pd.DataFrame) -> tuple[MLArtifacts, dict]:
    """
    Train 1 model dự đoán Amount(t) từ lag1..lag3 + Profit + Quantity (cùng tháng t).
    """
    df = add_lags(monthly, 3).dropna().copy()

    feature_cols = ["lag1", "lag2", "lag3", "Profit", "Quantity"]
    X = df[feature_cols].values
    y = df["Amount"].values

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1
    )
    model.fit(X, y)

    # quick eval (train-split theo thời gian đơn giản: dùng 20% cuối làm test)
    df_sorted = df.sort_values(["Sub-Category", "YearMonth"])
    cut = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:cut]
    test_df = df_sorted.iloc[cut:]

    model2 = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )
    model2.fit(train_df[feature_cols].values, train_df["Amount"].values)
    pred = model2.predict(test_df[feature_cols].values)
    mae = mean_absolute_error(test_df["Amount"].values, pred)

    artifacts = MLArtifacts(model=model, feature_cols=feature_cols)
    metrics = {"MAE_holdout": float(mae), "n_rows_trainable": int(len(df))}
    return artifacts, metrics

def predict_next_for_each_subcategory(monthly: pd.DataFrame, artifacts: MLArtifacts) -> pd.DataFrame:
    """
    Lấy 3 tháng gần nhất của mỗi Sub-Category để dự đoán tháng kế tiếp.
    """
    df = monthly.copy().sort_values(["Sub-Category", "YearMonth"])
    out_rows = []

    for subcat, g in df.groupby("Sub-Category"):
        g = g.sort_values("YearMonth")
        if len(g) < 4:
            continue

        last = g.iloc[-1]
        lag1 = g.iloc[-1]["Amount"]
        lag2 = g.iloc[-2]["Amount"]
        lag3 = g.iloc[-3]["Amount"]

        feat = np.array([[lag1, lag2, lag3, last["Profit"], last["Quantity"]]])
        next_pred = float(artifacts.model.predict(feat)[0])

        out_rows.append({"Sub-Category": subcat, "Pred_Amount_1m": next_pred})

    return pd.DataFrame(out_rows)

def predict_3_months_ahead(monthly: pd.DataFrame, artifacts: MLArtifacts) -> pd.DataFrame:
    """
    Dự đoán 3 tháng tới bằng cách roll-forward:
    - tháng+1: dùng lag thật
    - tháng+2: dùng pred của tháng+1 làm lag1...
    """
    df = monthly.copy().sort_values(["Sub-Category", "YearMonth"])
    rows = []

    for subcat, g in df.groupby("Sub-Category"):
        g = g.sort_values("YearMonth")
        if len(g) < 4:
            continue

        # last known
        last_profit = float(g.iloc[-1]["Profit"])
        last_qty = float(g.iloc[-1]["Quantity"])

        lag1 = float(g.iloc[-1]["Amount"])
        lag2 = float(g.iloc[-2]["Amount"])
        lag3 = float(g.iloc[-3]["Amount"])

        preds = []
        for _ in range(3):
            feat = np.array([[lag1, lag2, lag3, last_profit, last_qty]])
            p = float(artifacts.model.predict(feat)[0])
            preds.append(p)
            # roll forward lags
            lag3, lag2, lag1 = lag2, lag1, p

        rows.append({
            "Sub-Category": subcat,
            "Pred_Amount_1m": preds[0],
            "Pred_Amount_3m_sum": sum(preds),
            "Pred_Amount_3m_avg": sum(preds) / 3.0
        })

    return pd.DataFrame(rows)

def save_model(artifacts: MLArtifacts, path: str) -> None:
    joblib.dump({"model": artifacts.model, "feature_cols": artifacts.feature_cols}, path)

def load_model(path: str) -> MLArtifacts:
    obj = joblib.load(path)
    return MLArtifacts(model=obj["model"], feature_cols=obj["feature_cols"])