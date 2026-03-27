from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

API_BASE = "http://127.0.0.1:8000"
CRITERIA_ORDER = ["Amount_total", "Profit_total", "Quantity_total", "Stability"]
CRITERIA_LABELS = {
    "Amount_total": "Doanh thu",
    "Profit_total": "Lợi nhuận",
    "Quantity_total": "Sản lượng",
    "Stability": "Độ ổn định",
}
RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def get_meta() -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}/meta", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {
            "subcategories": [],
            "months": {"all": [], "min": None, "max": None},
            "criteria_order": CRITERIA_ORDER,
            "pairwise_default": [
                [1, 3, 2, 5],
                [1 / 3, 1, 1 / 2, 3],
                [1 / 2, 2, 1, 4],
                [1 / 5, 1 / 3, 1 / 4, 1],
            ],
            "horizons": ["1m", "3m"],
        }


def _float_or_default(value: Optional[str], default: float = 1.0) -> float:
    if value is None:
        return default
    raw = str(value).strip()
    if raw == "":
        return default
    try:
        out = float(raw)
        if out <= 0:
            return default
        return out
    except Exception:
        return default


def parse_pairwise_matrix(form_data: Any, default_matrix: List[List[float]]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for i in range(4):
        row: List[float] = []
        for j in range(4):
            default = float(default_matrix[i][j])
            value = _float_or_default(form_data.get(f"m_{i}_{j}"), default)
            if i == j:
                value = 1.0
            row.append(value)
        matrix.append(row)
    return matrix


def ahp_details(matrix: List[List[float]] | np.ndarray) -> Dict[str, Any]:
    A = np.array(matrix, dtype=float)
    n = int(A.shape[0])

    # Đúng theo file PDF: chuẩn hóa theo cột rồi lấy trung bình theo hàng
    col_sums = A.sum(axis=0)
    normalized = np.divide(A, col_sums, out=np.zeros_like(A), where=col_sums != 0)
    w = normalized.mean(axis=1)
    w = np.ones(n) / n if float(w.sum()) == 0 else w / w.sum()

    # Kiểm tra nhất quán
    weighted_sum = A @ w
    consistency_vector = np.divide(weighted_sum, w, out=np.zeros_like(weighted_sum), where=w != 0)
    lambda_max = float(consistency_vector.mean()) if n > 0 else 0.0
    ci = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    ri = RI_TABLE.get(n, 1.49)
    cr = (ci / ri) if ri > 0 else 0.0

    return {
        "pairwise": A.round(6).tolist(),
        "column_sums": col_sums.round(6).tolist(),
        "normalized": normalized.round(6).tolist(),
        "weights": w.round(6).tolist(),
        "weighted_sum": weighted_sum.round(6).tolist(),
        "consistency_vector": consistency_vector.round(6).tolist(),
        "lambda_max": float(lambda_max),
        "CI": float(ci),
        "CR": float(cr),
    }


def build_pairwise_from_values(alts: List[str], vals: List[float]) -> List[List[float]]:
    n = len(alts)
    m = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            den = float(vals[j])
            m[i, j] = 0.0 if abs(den) < 1e-12 else float(vals[i]) / den
    return m.tolist()


def build_alternative_analyses(results: List[Dict[str, Any]], max_items: int = 4) -> Dict[str, Any]:
    if not results:
        return {"analyses": [], "summary_rows": [], "alternatives": []}

    rows = sorted(results, key=lambda x: x.get("Rank", 999999))[:max_items]
    alternatives = [str(r["SubCategory"]) for r in rows]
    criteria_map = {
        "Amount_total": "Doanh thu",
        "Profit_total": "Lợi nhuận",
        "Quantity_total": "Sản lượng",
        "Stability": "Độ ổn định",
    }

    analyses: List[Dict[str, Any]] = []
    local_vectors: Dict[str, List[float]] = {}
    for crit, label in criteria_map.items():
        values = [float(r.get(crit, 0.0) or 0.0) for r in rows]
        pairwise = build_pairwise_from_values(alternatives, values)
        detail = ahp_details(pairwise)
        local_vectors[crit] = detail["weights"]
        analyses.append({
            "criterion": crit,
            "label": label,
            "alternatives": alternatives,
            "source_values": values,
            **detail,
        })

    summary_rows: List[Dict[str, Any]] = []
    for idx, alt in enumerate(alternatives):
        row: Dict[str, Any] = {"SubCategory": alt}
        for crit in CRITERIA_ORDER:
            row[crit] = local_vectors[crit][idx]
        summary_rows.append(row)

    return {"analyses": analyses, "summary_rows": summary_rows, "alternatives": alternatives}


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    meta = get_meta()
    months = meta.get("months", {}).get("all", [])
    subcategories = meta.get("subcategories", [])
    default_matrix = meta.get("pairwise_default") or [
        [1, 3, 2, 5],
        [1 / 3, 1, 1 / 2, 3],
        [1 / 2, 2, 1, 4],
        [1 / 5, 1 / 3, 1 / 4, 1],
    ]

    form_data: Dict[str, Any] = {
        "start_month": months[0] if months else "",
        "end_month": months[-1] if months else "",
        "top_n": 10,
        "horizon": "1m",
        "use_subcat_filter": False,
        "subcategories": [],
        "matrix": default_matrix,
    }

    error: Optional[str] = None
    backend_error: Optional[str] = None
    criteria_detail: Optional[Dict[str, Any]] = None
    weights_rows: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    alt_payload: Dict[str, Any] = {"analyses": [], "summary_rows": [], "alternatives": []}
    best_option: Optional[Dict[str, Any]] = None
    ml_note: Optional[str] = None
    ml_metrics: Dict[str, Any] = {}
    charts: Dict[str, List[Any]] = {"score_labels": [], "score_values": [], "pred_labels": [], "pred_values": []}
    score_display_map: Dict[str, float] = {}
    current_products: List[str] = subcategories[:]
    criteria_status = {"ok": False, "label": "Chưa đánh giá"}

    if request.method == "POST":
        selected_subcategories = request.form.getlist("subcategories")
        use_subcat_filter = request.form.get("use_subcat_filter") == "on"
        pairwise_matrix = parse_pairwise_matrix(request.form, default_matrix)
        form_data = {
            "start_month": request.form.get("start_month", ""),
            "end_month": request.form.get("end_month", ""),
            "top_n": int(request.form.get("top_n", 10) or 10),
            "horizon": request.form.get("horizon", "1m"),
            "use_subcat_filter": use_subcat_filter,
            "subcategories": selected_subcategories,
            "matrix": pairwise_matrix,
        }
        current_products = selected_subcategories if (use_subcat_filter and selected_subcategories) else subcategories[:]

        criteria_detail = ahp_details(pairwise_matrix)
        criteria_status = {
            "ok": criteria_detail["CR"] < 0.1,
            "label": "Đạt (CR < 0.1)" if criteria_detail["CR"] < 0.1 else "Chưa đạt (CR ≥ 0.1)",
        }
        weights_rows = [
            {"key": key, "label": CRITERIA_LABELS[key], "weight": criteria_detail["weights"][idx]}
            for idx, key in enumerate(CRITERIA_ORDER)
        ]

        if criteria_detail["CR"] >= 0.1:
            error = "Ma trận so sánh cặp chưa nhất quán (CR ≥ 0.1). Bạn cần nhập lại trước khi chạy xếp hạng."
        else:
            payload: Dict[str, Any] = {
                "start_month": form_data["start_month"],
                "end_month": form_data["end_month"],
                "top_n": form_data["top_n"],
                "horizon": form_data["horizon"],
                "pairwise_matrix": pairwise_matrix,
            }
            if use_subcat_filter and selected_subcategories:
                payload["subcategories"] = selected_subcategories
            try:
                r = requests.post(f"{API_BASE}/rank", json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
                ml = data.get("ml", {})
                ml_note = ml.get("note")
                ml_metrics = ml.get("metrics", {})
                if results:
                    alt_payload = build_alternative_analyses(results, max_items=len(results))
                    best_option = sorted(results, key=lambda x: x.get("Rank", 999999))[0]

                    # Tính lại Score hiển thị đúng theo file AHP:
                    # Score(i) = sum( w_k * p_i,k )
                    weight_map = {
                        key: float(criteria_detail["weights"][idx])
                        for idx, key in enumerate(CRITERIA_ORDER)
                    }

                    for row in alt_payload.get("summary_rows", []):
                        score_display_map[str(row["SubCategory"])] = sum(
                            float(row.get(key, 0.0) or 0.0) * weight_map[key]
                            for key in CRITERIA_ORDER
                        )

                    pred_key = "Pred_Amount_1m" if form_data["horizon"] == "1m" else "Pred_Amount_3m_avg"
                    charts = {
                        "score_labels": [str(r["SubCategory"]) for r in results],
                        "score_values": [
                            score_display_map.get(str(r["SubCategory"]), float(r.get("Score", 0.0) or 0.0))
                            for r in results
                        ],
                        "pred_labels": [str(r["SubCategory"]) for r in results if r.get(pred_key) is not None],
                        "pred_values": [float(r.get(pred_key, 0.0) or 0.0) for r in results if r.get(pred_key) is not None],
                    }
            except Exception as exc:
                backend_error = f"Không gọi được backend xếp hạng: {exc}"

    return render_template(
        "index.html",
        months=months,
        subcategories=subcategories,
        current_products=current_products,
        criteria_keys=CRITERIA_ORDER,
        criteria_labels=CRITERIA_LABELS,
        criteria_vn_order=[CRITERIA_LABELS[k] for k in CRITERIA_ORDER],
        form_data=form_data,
        error=error,
        backend_error=backend_error,
        criteria_detail=criteria_detail,
        criteria_status=criteria_status,
        weights_rows=weights_rows,
        results=results,
        alternative_analyses=alt_payload.get("analyses", []),
        alternative_summary_rows=alt_payload.get("summary_rows", []),
        best_option=best_option,
        ml_note=ml_note,
        ml_metrics=ml_metrics,
        charts=charts,
        score_display_map=score_display_map,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
