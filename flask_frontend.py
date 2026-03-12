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
            "months": {
                "all": [],
                "min": None,
                "max": None
            },
            "pairwise_default": [
                [1, 3, 2, 5],
                [1/3, 1, 1/2, 3],
                [1/2, 2, 1, 4],
                [1/5, 1/3, 1/4, 1]
            ],
            "horizons": ["1m", "3m"]
        }


@app.route("/", methods=["GET", "POST"])
def index():
    meta = get_meta()
    subcategories = meta.get("subcategories", [])
    months = meta.get("months", {}).get("all", [])
    default_matrix = meta.get("pairwise_default", [
        [1, 3, 2, 5],
        [1/3, 1, 1/2, 3],
        [1/2, 2, 1, 4],
        [1/5, 1/3, 1/4, 1]
    ])

    results = []
    weights = {}
    consistency = {}
    error = None
    best_option = None
    cr_status = {
        "label": "Chưa đánh giá",
        "class": "neutral"
    }

    form_data = {
        "start_month": months[0] if months else "",
        "end_month": months[-1] if months else "",
        "top_n": 10,
        "horizon": "1m",
        "subcategories": [],
        "matrix": default_matrix
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
                    value = request.form.get(f"m_{i}_{j}", "1").strip()
                    if value == "":
                        value = "1"
                    row.append(float(value))
                pairwise_matrix.append(row)

            form_data = {
                "start_month": start_month,
                "end_month": end_month,
                "top_n": top_n,
                "horizon": horizon,
                "subcategories": selected_subcategories,
                "matrix": pairwise_matrix
            }

            payload = {
                "start_month": start_month,
                "end_month": end_month,
                "top_n": top_n,
                "horizon": horizon,
                "subcategories": selected_subcategories if selected_subcategories else None,
                "pairwise_matrix": pairwise_matrix
            }

            r = requests.post(f"{API_BASE}/rank", json=payload, timeout=120)

            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                ahp_data = data.get("ahp", {})
                weights = ahp_data.get("weights", {})
                consistency = ahp_data.get("consistency", {})

                if results:
                    best_option = results[0]

                cr_value = consistency.get("CR")
                if cr_value is not None:
                    if cr_value < 0.1:
                        cr_status = {
                            "label": "Ma trận nhất quán, có thể chấp nhận",
                            "class": "good"
                        }
                    else:
                        cr_status = {
                            "label": "Ma trận chưa nhất quán, nên nhập lại",
                            "class": "bad"
                        }
            else:
                error = f"Backend lỗi {r.status_code}: {r.text}"

        except Exception as e:
            error = f"Lỗi xử lý: {str(e)}"

    return render_template(
        "index.html",
        months=months,
        subcategories=subcategories,
        criteria_labels=CRITERIA_LABELS,
        form_data=form_data,
        results=results,
        weights=weights,
        consistency=consistency,
        error=error,
        best_option=best_option,
        cr_status=cr_status
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)