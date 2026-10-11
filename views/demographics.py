import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_demographics(df_raw):
    st.title("🛠️ Chương 2: Tiền Xử Lý & Làm Sạch Dữ Liệu")
    st.caption(
        "Khám phá thực trạng dữ liệu thô, xem mã nguồn xử lý và thực thi làm sạch dữ liệu trực tiếp."
    )
    st.markdown("---")

    # ----------------------------------------------------
    # BƯỚC 1: HIỂN THỊ THỰC TRẠNG DỮ LIỆU THÔ & KHUYẾT THIẾU
    # ----------------------------------------------------
    st.subheader("1. Kiểm tra Dữ liệu Khuyết (Missing Values)")

    # Tính toán số lượng và tỷ lệ giá trị khuyết từ df_raw gốc
    missing_count = df_raw.isnull().sum()
    missing_percent = (df_raw.isnull().sum() / len(df_raw)) * 100

    missing_df = pd.DataFrame(
        {
            "Cột Dữ Liệu": df_raw.columns,
            "Số Lượng Khuyết (NaN)": missing_count.values,
            "Tỷ Lệ Khuyết (%)": missing_percent.values.round(2),
        }
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("📊 **Bảng thống kê giá trị khuyết:**")
        st.dataframe(missing_df, use_container_width=True)

    with col2:
        st.write("📈 **Biểu đồ tỷ lệ dữ liệu bị khuyết:**")
        fig_missing = px.bar(
            missing_df[missing_df["Số Lượng Khuyết (NaN)"] > 0],
            x="Cột Dữ Liệu",
            y="Số Lượng Khuyết (NaN)",
            text="Số Lượng Khuyết (NaN)",
            title="Các cột có chứa giá trị NaN",
            color_discrete_sequence=["#EF553B"],
        )
        if (
            missing_df["Số Lượng Khuyết (NaN)"].sum() == 0
        ):  # Nếu không có lỗi NaN
            st.info("🎉 Dữ liệu hiện tại không có giá trị khuyết (NaN) nào!")
        else:
            st.plotly_chart(fig_missing, use_container_width=True)

    st.info(
        "ℹ️ Các ô khuyết ở cột *Sleep Disorder* thực chất là chữ `None` (không mắc rối loạn) "
        "trong file CSV, bị `pd.read_csv` đọc thành NaN. Vì vậy điền lại bằng `'None'` là hợp lý, "
        "không phải suy đoán giá trị thiếu."
    )

    # Kiểm tra hồ sơ trùng lặp (bỏ qua cột Person ID)
    profile_cols = [c for c in df_raw.columns if c != "Person ID"]
    n_profiles = len(df_raw[profile_cols].drop_duplicates())
    st.warning(
        f"⚠️ **Dữ liệu trùng lặp:** {len(df_raw)} dòng nhưng chỉ có **{n_profiles} hồ sơ khác nhau** "
        "(các dòng còn lại giống hệt nhau, chỉ khác *Person ID*). Các chương biểu đồ vẫn dùng đủ "
        f"{len(df_raw)} dòng; riêng Chương 7 chia Train/Test theo hồ sơ để bản sao không nằm ở cả hai tập."
    )

    st.markdown("---")

    # ----------------------------------------------------
    # BƯỚC 2: CỬA SỔ HIỂN THỊ CODE TIỀN XỬ LÝ
    # ----------------------------------------------------
    st.subheader("2. Mã Nguồn Xử Lý Dữ Liệu (Code Snippet)")
    st.markdown(
        "Đoạn mã bên dưới sẽ thực hiện: Điền `None` cho các ô thiếu ở cột *Sleep Disorder*, chuẩn hóa cột *BMI Category*, và tách cột *Blood Pressure* thành *Systolic_BP* & *Diastolic_BP*."
    )

    # Cửa sổ hiển thị code Python
    code_snippet = """
# 1. Xử lý giá trị khuyết ở cột Sleep Disorder
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# 2. Chuẩn hóa tên nhóm BMI (Gộp 'Normal Weight' thành 'Normal')
df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})

# 3. Tách cột Blood Pressure thành 2 chỉ số riêng biệt (Tâm thu / Tâm trương)
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

# 4. Phân nhóm độ tuổi
df['Age_Group'] = pd.cut(
    df['Age'], 
    bins=[20, 35, 50, 65], 
    labels=['20-35 (Trẻ)', '36-50 (Trung niên)', '51-65 (Lớn tuổi)']
)
"""
    st.code(code_snippet, language="python")

    st.markdown("---")

    # ----------------------------------------------------
    # BƯỚC 3: NÚT KÍCH HOẠT XỬ LÝ & KẾT QUẢ
    # ----------------------------------------------------
    st.subheader("3. Thực Thi Xử Lý Dữ Liệu")

    # Sử dụng Session State để lưu trạng thái đã bấm nút hay chưa
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = False

    # Nút bấm xử lý
    if st.button("⚙️ Bắt đầu Xử lý Dữ liệu", type="primary"):
        st.session_state.is_processed = True

    # Hiển thị thông báo và kết quả dựa trên trạng thái
    if st.session_state.is_processed:
        # Thực hiện các bước tiền xử lý
        df_clean = df_raw.copy()
        df_clean["Sleep Disorder"] = df_clean["Sleep Disorder"].fillna("None")
        df_clean["BMI Category"] = df_clean["BMI Category"].replace(
            {"Normal Weight": "Normal"}
        )
        if "Blood Pressure" in df_clean.columns:
            df_clean[["Systolic_BP", "Diastolic_BP"]] = (
                df_clean["Blood Pressure"].str.split("/", expand=True).astype(int)
            )

        # Thông báo thành công
        st.success(
            "✅ **Đã xử lý dữ liệu thành công!** Các giá trị NaN đã được thay thế và dữ liệu đã sẵn sàng cho phân tích."
        )

        # Show kết quả so sánh
        tab_before, tab_after = st.tabs(
            ["❌ Dữ liệu Thô (Trước xử lý)", "✅ Dữ liệu Sạch (Sau xử lý)"]
        )

        with tab_before:
            st.write("Dữ liệu gốc chưa qua làm sạch:")
            st.dataframe(df_raw.head(10), use_container_width=True)

        with tab_after:
            st.write("Dữ liệu đã được tiền xử lý thành công (Đã thêm/sửa các cột):")
            st.dataframe(df_clean.head(10), use_container_width=True)

            # Đếm lại số lượng khuyết sau xử lý
            st.caption(
                f"📌 Số lượng giá trị NaN còn lại ở cột 'Sleep Disorder': **{df_clean['Sleep Disorder'].isnull().sum()}**"
            )
    else:
        st.info("👉 Nhấn vào nút **'Bắt đầu Xử lý Dữ liệu'** ở trên để chạy mã làm sạch dữ liệu.")