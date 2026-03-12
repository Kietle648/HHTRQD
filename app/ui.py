import requests
import streamlit as st
import pandas as pd
import numpy as np

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DSS AHP + ML", layout="wide", initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown(
    """
<style>
.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 4px;
}
.sub-title {
    color: #6b7280;
    font-size: 15px;
    margin-bottom: 18px;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
}
div[data-testid="stMetric"] {
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 12px;
    border-radius: 12px;
}
.section-card {
    padding: 14px 16px;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    background: #ffffff;
    margin-bottom: 14px;
}
.small-note {
    color: #6b7280;
    font-size: 13px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Hệ hỗ trợ ra quyết định lựa chọn sản phẩm kinh doanh chủ lực</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Kết hợp AHP để xác định trọng số tiêu chí và Machine Learning để dự đoán doanh thu tương lai.</div>',
    unsafe_allow_html=True,
)


# =========================
# Helpers
# =========================
def get_meta():
    r = requests.get(f"{API_BASE}/meta", timeout=30)
    r.raise_for_status()
    return r.json()


def post_rank(payload):
    r = requests.post(f"{API_BASE}/rank", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def style_ranking_table(df: pd.DataFrame):
    fmt = {}
    for col in df.columns:
        if col == "Score":
            fmt[col] = "{:.4f}"
        elif col in [
            "Amount_total",
            "Profit_total",
            "Quantity_total",
            "Pred_Amount_1m",
            "Pred_Amount_3m_avg",
        ]:
            fmt[col] = "{:,.2f}"
        elif col == "Stability":
            fmt[col] = "{:.4f}"
    return df.style.format(fmt, na_rep="-")


def style_weight_table(df: pd.DataFrame):
    return df.style.format({"Trọng số": "{:.4f}"})


def enforce_criteria_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ép ma trận tiêu chí:
    - đường chéo = 1
    - giá trị không hợp lệ hoặc <=0 -> thay bằng 1
    """
    M = df.copy()
    for c in M.columns:
        M[c] = pd.to_numeric(M[c], errors="coerce")
    M = M.fillna(1.0)

    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            if M.iat[i, j] <= 0:
                M.iat[i, j] = 1.0
    for i in range(n):
        M.iat[i, i] = 1.0
    return M


def style_criteria_matrix(df: pd.DataFrame):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    n = df.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                styles.iat[i, j] = "background-color: #ffe599; font-weight: 700;"
            else:
                styles.iat[i, j] = "background-color: #f8fafc;"
    return df.style.apply(lambda _: styles, axis=None).format("{:.4f}")


def build_pairwise_alternative_matrix(
    df: pd.DataFrame, alt_col: str, value_col: str
) -> pd.DataFrame:
    """
    Tạo ma trận so sánh cặp phương án theo 1 tiêu chí:
    A[i,j] = value_i / value_j
    """
    alts = df[alt_col].tolist()
    vals = df[value_col].astype(float).tolist()
    n = len(alts)

    matrix = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if abs(vals[j]) < 1e-12:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = vals[i] / vals[j]

    return pd.DataFrame(matrix, index=alts, columns=alts)


def style_pairwise_matrix(df: pd.DataFrame):
    return df.style.format("{:.6f}").background_gradient(cmap="Oranges")


# =========================
# Load meta
# =========================
with st.spinner("Đang tải metadata từ backend..."):
    try:
        meta = get_meta()
    except Exception as e:
        st.error(
            "Không gọi được backend. Hãy đảm bảo backend đang chạy:\n"
            "uvicorn app.api:app --reload --host 127.0.0.1 --port 8000\n\n"
            f"Chi tiết: {e}"
        )
        st.stop()

subcats = meta.get("subcategories", [])
months = meta.get("months", {}).get("all", [])
default_A = meta.get(
    "pairwise_default",
    [
        [1, 3, 2, 5],
        [1 / 3, 1, 1 / 2, 3],
        [1 / 2, 2, 1, 4],
        [1 / 5, 1 / 3, 1 / 4, 1],
    ],
)

# ÉP THỨ TỰ TIÊU CHÍ CỐ ĐỊNH
criteria_order = ["Amount_total", "Profit_total", "Quantity_total", "Stability"]

criteria_name_map = {
    "Amount_total": "Doanh thu",
    "Profit_total": "Lợi nhuận",
    "Quantity_total": "Sản lượng",
    "Stability": "Độ ổn định",
}

criteria_vn_order = [criteria_name_map[c] for c in criteria_order]

# =========================
# Sidebar
# =========================
st.sidebar.header("Cấu hình hệ thống")

if months:
    start_month = st.sidebar.selectbox("Thời gian bắt đầu", months, index=0)
    end_month = st.sidebar.selectbox(
        "Thời gian kết thúc", months, index=len(months) - 1
    )
else:
    start_month = st.sidebar.text_input("Thời gian bắt đầu", value="2024-01")
    end_month = st.sidebar.text_input("Thời gian kết thúc", value="2024-12")

top_n = st.sidebar.slider("Số lượng phương án hiển thị (Top N)", 1, 50, 10, 1)

horizon_label = st.sidebar.selectbox(
    "Dự đoán doanh thu", ["1 tháng tới", "3 tháng tới"], index=0
)
horizon = "1m" if horizon_label.startswith("1") else "3m"

use_subcat_filter = st.sidebar.checkbox("Lọc theo sản phẩm", value=False)
selected_subcats = None
if use_subcat_filter:
    selected_subcats = st.sidebar.multiselect(
        "Chọn sản phẩm",
        subcats,
        default=subcats[:5] if len(subcats) >= 5 else subcats,
    )

st.sidebar.markdown("---")
st.sidebar.caption("Thứ tự tiêu chí AHP:")
st.sidebar.write(criteria_vn_order)

# =========================
# AHP matrix input
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Ma trận so sánh cặp tiêu chí (AHP)")
st.caption(
    "Bạn có thể chỉnh trực tiếp ma trận 4×4. "
    "Lưu ý: hệ thống sẽ tự động ép các ô đường chéo chính bằng 1 trước khi tính toán."
)

df_A = pd.DataFrame(default_A, columns=criteria_vn_order, index=criteria_vn_order)

if "ahp_editor_ver" not in st.session_state:
    st.session_state["ahp_editor_ver"] = 0

col_left, col_right = st.columns([2.2, 1])

with col_left:
    edited_A = st.data_editor(
        df_A,
        width="stretch",
        num_rows="fixed",
        key=f"ahp_matrix_editor_{st.session_state['ahp_editor_ver']}",
    )

with col_right:
    st.markdown("### Gợi ý")
    st.write("Thang AHP thường dùng: 1, 3, 5, 7, 9")
    st.write("CR < 0.1: ma trận nhất quán")
    st.write("Đường chéo chính sẽ luôn được hệ thống ép về 1.")
    if st.button("Reset về mặc định"):
        st.session_state["ahp_editor_ver"] += 1
        st.rerun()

criteria_matrix_used = enforce_criteria_matrix(edited_A)

# st.markdown("#### Ma trận tiêu chí sử dụng để tính toán")
# st.dataframe(style_criteria_matrix(criteria_matrix_used), width="stretch")
# st.markdown(
#     '<div class="small-note">Các ô trên đường chéo chính được tô màu vàng và luôn có giá trị bằng 1.</div>',
#     unsafe_allow_html=True,
# )
# st.markdown("</div>", unsafe_allow_html=True)

pairwise_matrix = criteria_matrix_used.values.tolist()

# =========================
# Run
# =========================
run = st.button("🚀 Chạy xếp hạng", type="primary")

if run:
    payload = {
        "start_month": start_month,
        "end_month": end_month,
        "top_n": top_n,
        "horizon": horizon,
        "pairwise_matrix": pairwise_matrix,
    }
    if selected_subcats:
        payload["subcategories"] = selected_subcats

    with st.spinner("Đang gọi backend và tính toán xếp hạng..."):
        try:
            resp = post_rank(payload)
        except requests.HTTPError as e:
            st.error(f"Backend trả lỗi: {e.response.status_code} - {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Lỗi gọi backend: {e}")
            st.stop()

    # =========================
    # Parse response
    # =========================
    ahp = resp.get("ahp", {})
    weights = ahp.get("weights", {})
    cons = ahp.get("consistency", {})
    ml = resp.get("ml", {})
    results = resp.get("results", [])

    if not results:
        st.warning("Không có kết quả.")
        st.stop()

    res_df = pd.DataFrame(results)

    # nếu có contrib
    if "contrib" in res_df.columns:
        contrib_df = pd.json_normalize(res_df["contrib"])
        contrib_df.columns = [f"contrib_{c}" for c in contrib_df.columns]
        res_df = pd.concat([res_df.drop(columns=["contrib"]), contrib_df], axis=1)

    # =========================
    # KPI row
    # =========================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Số phương án", len(res_df))
    k2.metric("CR", f"{cons.get('CR', 0):.4f}" if cons.get("CR") is not None else "N/A")
    k3.metric("CI", f"{cons.get('CI', 0):.4f}" if cons.get("CI") is not None else "N/A")
    k4.metric(
        "λ max",
        (
            f"{cons.get('lambda_max', 0):.4f}"
            if cons.get("lambda_max") is not None
            else "N/A"
        ),
    )

    # =========================
    # Tabs
    # =========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Bảng xếp hạng",
            "Trọng số tiêu chí",
            "Ma trận phương án",
            "ML dự đoán",
            "Biểu đồ",
        ]
    )

    # ---------- Tab 1: Ranking ----------
    with tab1:
        st.subheader("Bảng xếp hạng phương án")
        show_cols = [
            "Rank",
            "SubCategory",
            "Score",
            "Amount_total",
            "Profit_total",
            "Quantity_total",
            "Stability",
            "Pred_Amount_1m",
            "Pred_Amount_3m_avg",
        ]
        existing_cols = [c for c in show_cols if c in res_df.columns]
        st.dataframe(style_ranking_table(res_df[existing_cols]), width="stretch")

    # ---------- Tab 2: Criteria weights ----------
    with tab2:
        st.subheader("Trọng số tiêu chí theo AHP")

        wdf = pd.DataFrame(
            {
                "Tiêu chí": criteria_vn_order,
                "Ký hiệu": criteria_order,
                "Trọng số": [weights.get(c, 0.0) for c in criteria_order],
            }
        )
        st.dataframe(style_weight_table(wdf), width="stretch")

        chart_df = wdf.set_index("Tiêu chí")[["Trọng số"]]
        st.bar_chart(chart_df)

    # ---------- Tab 3: Alternative pairwise matrices ----------
    with tab3:
        st.subheader("Ma trận trọng số của các phương án theo từng tiêu chí")
        st.caption(
            "Mỗi tiêu chí được biểu diễn bằng một ma trận so sánh cặp giữa các phương án, "
            "với công thức A[i,j] = value_i / value_j."
        )

        alt_base = res_df[
            [
                "SubCategory",
                "Amount_total",
                "Profit_total",
                "Quantity_total",
                "Stability",
            ]
        ].copy()

        subt1, subt2, subt3, subt4 = st.tabs(
            ["Doanh thu", "Lợi nhuận", "Sản lượng", "Độ ổn định"]
        )

        with subt1:
            mat_amount = build_pairwise_alternative_matrix(
                alt_base, "SubCategory", "Amount_total"
            )
            st.dataframe(style_pairwise_matrix(mat_amount), width="stretch")

        with subt2:
            mat_profit = build_pairwise_alternative_matrix(
                alt_base, "SubCategory", "Profit_total"
            )
            st.dataframe(style_pairwise_matrix(mat_profit), width="stretch")

        with subt3:
            mat_quantity = build_pairwise_alternative_matrix(
                alt_base, "SubCategory", "Quantity_total"
            )
            st.dataframe(style_pairwise_matrix(mat_quantity), width="stretch")

        with subt4:
            mat_stability = build_pairwise_alternative_matrix(
                alt_base, "SubCategory", "Stability"
            )
            st.dataframe(style_pairwise_matrix(mat_stability), width="stretch")

    # ---------- Tab 4: ML ----------
    with tab4:
        st.subheader("Kết quả Machine Learning")
        st.write(ml.get("note", ""))
        st.json(ml.get("metrics", {}))

        pred_col = "Pred_Amount_1m" if horizon == "1m" else "Pred_Amount_3m_avg"
        if pred_col in res_df.columns:
            ml_show = res_df[["Rank", "SubCategory", pred_col]].copy()
            st.dataframe(ml_show.style.format({pred_col: "{:,.2f}"}), width="stretch")

    # ---------- Tab 5: Charts ----------
    with tab5:
        st.subheader("Biểu đồ phân tích")
        c1, c2 = st.columns(2)

        with c1:
            st.caption("Điểm tổng hợp theo phương án")
            st.bar_chart(res_df.set_index("SubCategory")["Score"])

        with c2:
            pred_col = "Pred_Amount_1m" if horizon == "1m" else "Pred_Amount_3m_avg"
            if pred_col in res_df.columns:
                st.caption("Doanh thu dự đoán")
                st.bar_chart(res_df.set_index("SubCategory")[pred_col])

st.markdown("---")
