import pandas as pd

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Chuẩn hoá: strip khoảng trắng tên cột
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_yearmonth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset của bạn có thể có cột 'Year-Month' hoặc 'Order Date'.
    - Nếu có 'Year-Month' -> dùng luôn
    - Nếu có 'Order Date'/'Order_Date' -> convert ra YearMonth
    """
    df = df.copy()

    # Trường hợp đã có Year-Month
    if "Year-Month" in df.columns:
        # cố gắng parse thành period month
        df["YearMonth"] = pd.to_datetime(df["Year-Month"]).dt.to_period("M").astype(str)
        return df

    # Trường hợp có Order Date
    date_cols = [c for c in df.columns if c.lower().replace(" ", "_") in ["order_date", "orderdate", "order date"]]
    if date_cols:
        col = date_cols[0]
        df["YearMonth"] = pd.to_datetime(df[col]).dt.to_period("M").astype(str)
        return df

    raise ValueError("Không tìm thấy cột thời gian: cần 'Year-Month' hoặc 'Order Date'.")

def validate_required_columns(df: pd.DataFrame) -> None:
    required = ["Sub-Category", "Amount", "Profit", "Quantity", "YearMonth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}. Hiện có: {list(df.columns)}")