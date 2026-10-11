import streamlit as st
from modules.data_loader import load_data, load_raw_data
from views.demographics import render_demographics
from views.evaluation import render_evaluation  # <--- Bổ sung import Chương 7
from views.explorer import render_explorer
from views.overview import render_overview
from views.regression import render_regression
from views.sleep_analysis import render_sleep_analysis
from views.vitals import render_vitals

# 1. Cấu hình Trang
st.set_page_config(
    page_title="Sleep & Lifestyle Dashboard", page_icon="🌙", layout="wide"
)

# 2. Load Dữ liệu
df_raw = load_data()

# 3. Sidebar Điều Hướng & Bộ Lọc
st.sidebar.title("📌 Danh Mục & Điều Hướng")
chapter = st.sidebar.radio(
    "Chọn Chương Phân Tích:",
    [
        "Chương 1: Tổng Quan & KPIs",
        "Chương 2: Tiền xử lý dữ liệu",
        "Chương 3: Trực quan hóa dữ liệu",
        "Chương 4: Công cụ trực quan",
        "Chương 5: Phân tích thăm dò",
        "Chương 6: Mô hình hồi quy",
        "Chương 7: Đánh giá mô hình",  # <--- Bổ sung Chương 7
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Bộ Lọc Dữ Liệu Chung")

all_genders = df_raw["Gender"].unique().tolist()
selected_genders = st.sidebar.multiselect(
    "Giới tính:", options=all_genders, default=all_genders
)

all_bmi = df_raw["BMI Category"].unique().tolist()
selected_bmi = st.sidebar.multiselect(
    "Nhóm BMI:", options=all_bmi, default=all_bmi
)

# Lọc dữ liệu
df = df_raw[
    (df_raw["Gender"].isin(selected_genders))
    & (df_raw["BMI Category"].isin(selected_bmi))
]

if df.empty:
    st.warning("⚠️ Không có dữ liệu phù hợp!")
    st.stop()

# 4. Router (Định tuyến)
if chapter == "Chương 1: Tổng Quan & KPIs":
    render_overview(df)

elif chapter == "Chương 2: Tiền xử lý dữ liệu":
    # Chương 2 cần dữ liệu thô (chưa fillna) để minh họa bước làm sạch
    render_demographics(load_raw_data())

elif chapter == "Chương 3: Trực quan hóa dữ liệu":
    render_sleep_analysis(df)

elif chapter == "Chương 4: Công cụ trực quan":
    render_vitals(df)

elif chapter == "Chương 5: Phân tích thăm dò":
    render_explorer(df)

elif chapter == "Chương 6: Mô hình hồi quy":
    render_regression(df)

elif chapter == "Chương 7: Đánh giá mô hình":  # <--- Bổ sung Route Chương 7
    render_evaluation(df)