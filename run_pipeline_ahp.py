import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

CSV_PATH = "data/raw/Sales Dataset.csv"

# =========================
# 1) AHP TRUE (pairwise -> weights + CR)
# =========================
RI_TABLE = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def ahp_weights(pairwise: np.ndarray):
    """
    Input: pairwise matrix A (n x n)
    Output: weights (n,), info dict with CR
    """
    A = np.array(pairwise, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Pairwise matrix phải là ma trận vuông (n x n).")

    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    lambda_max = float(eigvals.real[idx])

    w = eigvecs[:, idx].real
    w = w / w.sum()

    CI = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = (CI / RI) if RI > 1e-12 else 0.0

    return w.astype(float), {"lambda_max": lambda_max, "CI": float(CI), "CR": float(CR)}


# =========================
# 2) Data helpers
# =========================
def detect_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đảm bảo có đúng các cột:
    Sub-Category, Amount, Profit, Quantity, Year-Month hoặc Order Date
    Nếu dataset dùng Sales thay Amount -> map Sales -> Amount.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        lc = c.lower().strip()

        # Amount mapping (nhiều dataset dùng "Sales")
        if lc == "sales":
            rename_map[c] = "Amount"

        # sub-category mapping
        if lc in ["sub-category", "subcategory", "sub_category"]:
            rename_map[c] = "Sub-Category"

        # date mapping
        if lc in ["order date", "order_date", "orderdate", "date"]:
            rename_map[c] = "Order Date"

        if lc in ["year-month", "year_month", "yearmonth"]:
            rename_map[c] = "Year-Month"

    df = df.rename(columns=rename_map)
    return df


def ensure_yearmonth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Year-Month" in df.columns:
        df["YearMonth"] = pd.to_datetime(df["Year-Month"]).dt.to_period("M").astype(str)
        return df
    if "Order Date" in df.columns:
        df["YearMonth"] = pd.to_datetime(df["Order Date"]).dt.to_period("M").astype(str)
        return df
    raise ValueError(
        "Không tìm thấy cột thời gian. Cần 'Year-Month' hoặc 'Order Date'."
    )


def validate(df: pd.DataFrame):
    needed = ["Sub-Category", "Amount", "Profit", "Quantity", "YearMonth"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Thiếu cột: {miss}. Hiện có: {list(df.columns)}")


def make_monthly(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["Sub-Category", "YearMonth"], as_index=False)
        .agg({"Amount": "sum", "Profit": "sum", "Quantity": "sum"})
        .sort_values(["Sub-Category", "YearMonth"])
    )
    return monthly


# =========================
# 3) Criteria: Amount_total, Profit_total, Quantity_total, Stability
# =========================
def stability_from_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Stability dựa trên CV = std/mean theo Amount theo tháng.
    StabilityScore = 1/(CV + eps) => càng lớn càng ổn định.
    """
    eps = 1e-9
    g = monthly.groupby("Sub-Category")["Amount"]
    mean = g.mean()
    std = g.std(ddof=0)
    cv = std / (mean + eps)
    stability = 1.0 / (cv + eps)

    return pd.DataFrame({"Sub-Category": mean.index, "Stability": stability.values})


def criteria_table(df: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby("Sub-Category", as_index=False)
        .agg({"Amount": "sum", "Profit": "sum", "Quantity": "sum"})
        .rename(
            columns={
                "Amount": "Amount_total",
                "Profit": "Profit_total",
                "Quantity": "Quantity_total",
            }
        )
    )
    stab = stability_from_monthly(monthly)
    return base.merge(stab, on="Sub-Category", how="left")


# =========================
# 4) ML (THAM KHẢO): predict Amount 1 month / 3 months
#    -> KHÔNG đưa vào AHP score
# =========================
def add_lags(monthly: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    m = monthly.copy()
    for i in range(1, k + 1):
        m[f"lag{i}"] = m.groupby("Sub-Category")["Amount"].shift(i)
    return m


def train_model(monthly: pd.DataFrame):
    m = add_lags(monthly, 3).dropna().copy()
    feats = ["lag1", "lag2", "lag3", "Profit", "Quantity"]

    m = m.sort_values(["Sub-Category", "YearMonth"])
    cut = int(len(m) * 0.8)
    train, test = m.iloc[:cut], m.iloc[cut:]

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1
    )
    model.fit(train[feats].values, train["Amount"].values)

    pred = model.predict(test[feats].values)
    mae = mean_absolute_error(test["Amount"].values, pred)

    # fit full để predict
    model_full = RandomForestRegressor(
        n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1
    )
    model_full.fit(m[feats].values, m["Amount"].values)

    return model_full, feats, float(mae), int(len(m))


def predict_1m_3m(monthly: pd.DataFrame, model):
    df = monthly.sort_values(["Sub-Category", "YearMonth"]).copy()
    rows = []

    for subcat, g in df.groupby("Sub-Category"):
        g = g.sort_values("YearMonth")
        if len(g) < 4:
            continue

        last = g.iloc[-1]
        lag1 = float(g.iloc[-1]["Amount"])
        lag2 = float(g.iloc[-2]["Amount"])
        lag3 = float(g.iloc[-3]["Amount"])
        profit = float(last["Profit"])
        qty = float(last["Quantity"])

        preds = []
        for _ in range(3):
            X = np.array([[lag1, lag2, lag3, profit, qty]])
            p = float(model.predict(X)[0])
            preds.append(p)
            lag3, lag2, lag1 = lag2, lag1, p

        rows.append(
            {
                "Sub-Category": subcat,
                "Pred_Amount_1m": preds[0],
                "Pred_Amount_3m_avg": sum(preds) / 3.0,
            }
        )

    return pd.DataFrame(rows)


# =========================
# 5) Ranking: normalize + Score (CHỈ 4 TIÊU CHÍ)
# =========================
def minmax(s: pd.Series) -> pd.Series:
    mn, mx = float(s.min()), float(s.max())
    if abs(mx - mn) < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def rank_with_ahp_4(criteria: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Score chỉ dùng 4 tiêu chí:
    Amount_total, Profit_total, Quantity_total, Stability
    """
    df = criteria.copy()
    use_cols = ["Amount_total", "Profit_total", "Quantity_total", "Stability"]

    for c in use_cols:
        df[c + "_norm"] = minmax(df[c].fillna(0.0))

    score = 0.0
    for crit, w in weights.items():
        score = score + float(w) * df[crit + "_norm"]

    df["Score"] = score
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


# =========================
# 6) MAIN
# =========================
def main():
    # (A) Load + preprocess
    df = pd.read_csv(CSV_PATH)
    df = detect_and_rename_columns(df)
    df = ensure_yearmonth(df)
    validate(df)

    # (B) Monthly + criteria
    monthly = make_monthly(df)
    crit = criteria_table(df, monthly)

    # (C) ML chạy để tham khảo (không đưa vào AHP score)
    model, feats, mae, n_trainable = train_model(monthly)
    preds = predict_1m_3m(monthly, model)
    crit = crit.merge(preds, on="Sub-Category", how="left")

    print(
        f"ML metrics (tham khảo): {{'MAE_holdout': {mae}, 'n_rows_trainable': {n_trainable}}}"
    )

    # (D) AHP 4 tiêu chí: dùng MA TRẬN EXCEL CỦA BẠN
    # Thứ tự:
    # 1 Doanh thu (Amount_total)
    # 2 Lợi nhuận (Profit_total)
    # 3 Sản lượng (Quantity_total)
    # 4 Độ ổn định (Stability)
    criteria_order = ["Amount_total", "Profit_total", "Quantity_total", "Stability"]

    A = np.array(
        [
            [1, 3, 2, 5],
            [1 / 3, 1, 1 / 2, 3],
            [1 / 2, 2, 1, 4],
            [1 / 5, 1 / 3, 1 / 4, 1],
        ],
        dtype=float,
    )

    w, info = ahp_weights(A)
    print("AHP consistency:", info)
    if info["CR"] > 0.1:
        print("WARNING: CR > 0.1 => so sánh cặp chưa nhất quán, nên chỉnh lại ma trận.")

    weights = {criteria_order[i]: float(w[i]) for i in range(len(criteria_order))}
    print("AHP weights (4 tiêu chí):", weights)

    # (E) Ranking theo AHP 4 tiêu chí
    ranked = rank_with_ahp_4(crit, weights)

    # (F) Hiển thị Top 15 (ML chỉ để tham khảo nên vẫn show cột Pred_Amount_1m)
    cols_show = [
        "Rank",
        "Sub-Category",
        "Score",
        "Amount_total",
        "Profit_total",
        "Quantity_total",
        "Stability",
        "Pred_Amount_1m",
    ]
    print("\nTop ranked:")
    print(ranked[cols_show].head(15))

    # (G) Save outputs
    ranked.to_csv("outputs_top_ranked.csv", index=False)
    crit.to_csv("outputs_criteria_table.csv", index=False)
    print("\nSaved: outputs_top_ranked.csv, outputs_criteria_table.csv")


if __name__ == "__main__":
    main()
