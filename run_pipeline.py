from src.pipeline import run_pipeline
from src.config import TABLES_DIR, MODELS_DIR
import pandas as pd


def main():
    csv_path = "data/raw/Sales Dataset.csv"  # đổi tên file nếu bạn đặt khác

    result = run_pipeline(
        csv_path=csv_path,
        use_ml=True,
        top_n=15,
        save_model_path=str(MODELS_DIR / "rf_amount_next_month.joblib"),
    )

    ranked = result["ranked"]
    criteria = result["criteria"]

    ranked.to_csv(TABLES_DIR / "top_ranked_subcategories.csv", index=False)
    criteria.to_csv(TABLES_DIR / "criteria_table.csv", index=False)

    print("ML metrics:", result["ml_metrics"])
    print("\nTop ranked:")
    print(
        ranked[
            [
                "Rank",
                "Sub-Category",
                "Score",
                "Amount_total",
                "Profit_total",
                "Quantity_total",
                "Stability",
                "Pred_Amount_1m",
            ]
        ]
    )


if __name__ == "__main__":
    main()
