import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    # Đọc dữ liệu từ thư mục data
    df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

    # Xử lý dữ liệu khuyết & chuẩn hóa
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
    df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

    # Tách Huyết áp
    if "Blood Pressure" in df.columns:
        df[["Systolic_BP", "Diastolic_BP"]] = (
            df["Blood Pressure"].str.split("/", expand=True).astype(int)
        )

    # Chia nhóm tuổi
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[20, 35, 50, 65],
        labels=["20-35 (Trẻ)", "36-50 (Trung niên)", "51-65 (Lớn tuổi)"],
    )

    return df