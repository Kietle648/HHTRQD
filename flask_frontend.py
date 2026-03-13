from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_BASE = "http://127.0.0.1:8000"

CRITERIA_LABELS = [
    ("Amount_total", "Doanh thu"),
    ("Profit_total", "Lợi nhuận"),
    ("Quantity_total", "Sản lượng"),
    ("Stability", "Độ ổn định"),
]


def get_meta():
    try:
        r = requests.get(f"{API_BASE}/meta", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {
            "subcategories": [],
            "months": {"all": [], "min": None, "max": None},
            "pairwise_default": [
                [1, 3, 2, 5],
                [1 / 3, 1, 1 / 2, 3],
                [1 / 2, 2, 1, 4],
                [1 / 5, 1 / 3, 1 / 4, 1],
            ],
            "horizons": ["1m", "3m"],
        }


def safe_float(value, default=1.0):
    try:
        out = float(value)
        if out <= 0:
            return default
        return out
    except Exception:
        return default


def build_pairwise_alternative_matrix(results, value_key, max_items=5):
    rows = results[:max_items]
    labels = [row.get("SubCategory", "-") for row in rows]
    values = [safe_float(row.get(value_key, 0), 0.0) for row in rows]
    n = len(labels)
    matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            denominator = values[j]
            if abs(denominator) < 1e-12:
                row.append(0.0)
            else:
                row.append(values[i] / denominator)
        matrix.append(row)

    return {"labels": labels, "matrix": matrix}


def flatten_metrics(data, prefix=""):
    rows = []
    if not isinstance(data, dict):
        return rows

    for key, value in data.items():
        label = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_metrics(value, f"{label}."))
        else:
            rows.append({"label": label, "value": value})
    return rows


def build_bar_rows(items, label_key, value_key, limit=None):
    rows = items[:limit] if limit else items
    cleaned = []
    max_value = 0.0

    for row in rows:
        value = row.get(value_key)
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        if numeric > max_value:
            max_value = numeric
        cleaned.append({"label": row.get(label_key, "-"), "value": numeric})

    for row in cleaned:
        row["width"] = 0 if max_value <= 0 else round((row["value"] / max_value) * 100, 1)
    return cleaned


@app.route("/", methods=["GET", "POST"])
def index():
    meta = get_meta()
    subcategories = meta.get("subcategories", [])
    months = meta.get("months", {}).get("all", [])
    default_matrix = meta.get(
        "pairwise_default",
        [
            [1, 3, 2, 5],
            [1 / 3, 1, 1 / 2, 3],
            [1 / 2, 2, 1, 4],
            [1 / 5, 1 / 3, 1 / 4, 1],
        ],
    )

    results = []
    weights = {}
    consistency = {}
    ml_info = {}
    ml_metrics = []
    error = None
    best_option = None
    matrix_preview_count = 0
    alternative_matrices = {}
    score_bars = []
    prediction_bars = []
    active_prediction_label = "Dự đoán 1 tháng"

    cr_status = {"label": "Chưa đánh giá", "class": "neutral"}

    form_data = {
        "start_month": months[0] if months else "",
        "end_month": months[-1] if months else "",
        "top_n": 10,
        "horizon": "1m",
        "subcategories": [],
        "matrix": default_matrix,
    }

    if request.method == "POST":
        try:
            start_month = request.form.get("start_month", "")
            end_month = request.form.get("end_month", "")
            top_n = int(request.form.get("top_n", 10))
            horizon = request.form.get("horizon", "1m")
            selected_subcategories = request.form.getlist("subcategories")

            pairwise_matrix = []
            for i in range(4):
                row = []
                for j in range(4):
                    if i == j:
                        row.append(1.0)
                        continue
                    value = request.form.get(f"m_{i}_{j}", "1").strip()
                    row.append(safe_float(value, 1.0))
                pairwise_matrix.append(row)

            form_data = {
                "start_month": start_month,
                "end_month": end_month,
                "top_n": top_n,
                "horizon": horizon,
                "subcategories": selected_subcategories,
                "matrix": pairwise_matrix,
            }

            payload = {
                "start_month": start_month,
                "end_month": end_month,
                "top_n": top_n,
                "horizon": horizon,
                "subcategories": selected_subcategories if selected_subcategories else None,
                "pairwise_matrix": pairwise_matrix,
            }

            r = requests.post(f"{API_BASE}/rank", json=payload, timeout=120)

            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                ahp_data = data.get("ahp", {})
                weights = ahp_data.get("weights", {})
                consistency = ahp_data.get("consistency", {})
                ml_info = data.get("ml", {})
                ml_metrics = flatten_metrics(ml_info.get("metrics", {}))

                if results:
                    best_option = results[0]
                    matrix_preview_count = min(len(results), 5)
                    alternative_matrices = {
                        "Amount_total": build_pairwise_alternative_matrix(results, "Amount_total", matrix_preview_count),
                        "Profit_total": build_pairwise_alternative_matrix(results, "Profit_total", matrix_preview_count),
                        "Quantity_total": build_pairwise_alternative_matrix(results, "Quantity_total", matrix_preview_count),
                        "Stability": build_pairwise_alternative_matrix(results, "Stability", matrix_preview_count),
                    }
                    score_bars = build_bar_rows(results, "SubCategory", "Score", limit=10)
                    pred_key = "Pred_Amount_1m" if horizon == "1m" else "Pred_Amount_3m_avg"
                    active_prediction_label = "Dự đoán 1 tháng" if horizon == "1m" else "Dự đoán trung bình 3 tháng"
                    prediction_bars = [
                        row for row in build_bar_rows(results, "SubCategory", pred_key, limit=10) if row["value"] > 0
                    ]

                cr_value = consistency.get("CR")
                if cr_value is not None:
                    if cr_value < 0.1:
                        cr_status = {
                            "label": "Ma trận nhất quán, có thể chấp nhận.",
                            "class": "good",
                        }
                    else:
                        cr_status = {
                            "label": "Ma trận chưa nhất quán, nên điều chỉnh lại các so sánh cặp.",
                            "class": "bad",
                        }
            else:
                error = f"Backend lỗi {r.status_code}: {r.text}"

        except Exception as exc:
            error = f"Lỗi xử lý: {str(exc)}"

    weight_rows = [
        {
            "code": code,
            "label": label,
            "value": float(weights.get(code, 0.0)) if weights else 0.0,
        }
        for code, label in CRITERIA_LABELS
    ]

    return render_template(
        "index.html",
        months=months,
        subcategories=subcategories,
        criteria_labels=CRITERIA_LABELS,
        form_data=form_data,
        results=results,
        weights=weights,
        weight_rows=weight_rows,
        consistency=consistency,
        ml_info=ml_info,
        ml_metrics=ml_metrics,
        error=error,
        best_option=best_option,
        cr_status=cr_status,
        alternative_matrices=alternative_matrices,
        matrix_preview_count=matrix_preview_count,
        score_bars=score_bars,
        prediction_bars=prediction_bars,
        active_prediction_label=active_prediction_label,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
