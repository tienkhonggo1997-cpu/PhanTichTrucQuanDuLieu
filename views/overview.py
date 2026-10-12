import streamlit as st
import pandas as pd


def render_overview(df: pd.DataFrame = None):
    st.title("📋 TRANG TỔNG QUAN (OVERVIEW)")
    st.caption(
        "Bức tranh tổng thể về bộ dữ liệu, quy trình tiền xử lý, danh mục trực quan hóa và phương pháp phân tích.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # 1. Tóm tắt bộ dữ liệu (Dataset Summary)
    # -------------------------------------------------------------------------
    st.header("1. Tóm Tắt Bộ Dữ Liệu (Dataset Summary)")

    col1, col2, col3, col4 = st.columns(4)
    total_rows = len(df) if df is not None else 374
    total_cols = len(df.columns) if df is not None else 13

    col1.metric("Tên bộ dữ liệu", "Sleep Health")
    col2.metric("Số lượng mẫu (Rows)", f"{total_rows} mẫu")
    col3.metric("Số thuộc tính (Columns)", f"{total_cols} cột")
    col4.metric("Chủ đề", "Sức khỏe & Lối sống")

    st.markdown(
        """
        - **Nguồn dữ liệu:** `Sleep_health_and_lifestyle_dataset.csv`
        - **Nội dung chính:** Thu thập thông tin cá nhân, chỉ số sức khỏe (*huyết áp, nhịp tim, BMI, chất lượng giấc ngủ*) và thói quen sinh hoạt (*hoạt động thể chất, mức độ căng thẳng, số bước chân hàng ngày*).
        """
    )
    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Thống kê tiền xử lý dữ liệu (Data Preprocessing Summary)
    # -------------------------------------------------------------------------
    st.header("2. Thống Kê Tiền Xử Lý Dữ Liệu (Data Preprocessing Summary)")

    preprocessing_data = [
        {
            "Bước xử lý": "1. Giá trị khuyết (Missing Values)",
            "Thuộc tính tác động": "`Sleep Disorder`",
            "Mô tả phương pháp xử lý": "Ban đầu có 219 giá trị khuyết (NaN). Đền bù bằng nhãn 'None' (không mắc rối loạn giấc ngủ)."
        },
        {
            "Bước xử lý": "2. Chuẩn hóa định dạng (Formatting)",
            "Thuộc tính tác động": "`BMI Category`, `Blood Pressure`",
            "Mô tả phương pháp xử lý": "Đổi nhãn 'Normal Weight' thành 'Normal'. Tách 'Blood Pressure' ('Systolic/Diastolic') thành 2 cột số: `Systolic_BP` và `Diastolic_BP`."
        },
        {
            "Bước xử lý": "3. Chuẩn hóa dữ liệu (Scaling)",
            "Thuộc tính tác động": "`Daily Steps`, `Age`",
            "Mô tả phương pháp xử lý": "Áp dụng Min-Max Scaling cho `Daily Steps` -> `Daily_Steps_MinMax`. Áp dụng Z-Score Normalization cho `Age` -> `Age_Zscore`."
        },
        {
            "Bước xử lý": "4. Gom nhóm dữ liệu (Binning)",
            "Thuộc tính tác động": "`Sleep Duration`",
            "Mô tả phương pháp xử lý": "Chia `Sleep Duration` thành 3 khoảng bằng `np.linspace` -> `Sleep_Duration_Bin` với các nhãn: ['Nghỉ ngơi ít', 'Trung bình', 'Nghỉ ngơi nhiều']."
        },
        {
            "Bước xử lý": "5. Mã hóa dữ liệu (Encoding)",
            "Thuộc tính tác động": "`BMI Category`, `Gender`",
            "Mô tả phương pháp xử lý": "Ordinal Mapping cho `BMI Category`: Normal: 0, Overweight: 1, Obese: 2 -> `BMI_Encoded`. One-Hot Encoding cho `Gender` -> `Gender_Female`, `Gender_Male`."
        }
    ]

    st.table(pd.DataFrame(preprocessing_data))
    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. Danh mục Trực quan hóa & Công cụ (Visualization & Tools Catalog)
    # -------------------------------------------------------------------------
    st.header("3. Danh Mục Trực Quan Hóa & Công Cụ (Visualization Catalog)")

    st.markdown(
        """
        - **Thư viện & Công cụ áp dụng:** `Matplotlib`, `Seaborn`, `Plotly`, `Pandas`, `Streamlit`
        - **Danh mục các mô hình biểu đồ chính:**
            1. **Boxplot (Biểu đồ hộp):** Trực quan hóa phân bố thời lượng ngủ (`Sleep Duration`) theo từng nghề nghiệp (`Occupation`).
            2. **Histogram & KDE (Biểu đồ tần số & Đường mật độ):** Trực quan hóa phân bố tần suất nhịp tim (`Heart Rate`).
            3. **Countplot / Barplot (Biểu đồ cột):** Thống kê số lượng từng loại rối loạn giấc ngủ (`Sleep Disorder`) theo danh mục BMI (`BMI Category`).
            4. **Scatterplot (Biểu đồ tán sắc):** Thể hiện mối tương quan giữa Mức độ căng thẳng (`Stress Level`) và Chất lượng giấc ngủ (`Quality of Sleep`), phân loại theo Giới tính/Rối loạn giấc ngủ.
        """
    )
    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. Danh mục Phân tích thăm dò & Mô hình (Analysis & Models Catalog)
    # -------------------------------------------------------------------------
    st.header("4. Danh Mục Phân Tích Thăm Dò & Mô Hình (Analysis Catalog)")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("4.1. Phân Tích EDA")
        st.markdown(
            """
            - **Phân tích đơn biến:** Thống kê mô tả (Mean, Std, Min, Max), kiểm tra độ lệch (Skewness) & độ nhọn (Kurtosis).
            - **Phân tích hai biến:** Mối liên hệ giữa Hoạt động thể chất / Số bước chân với Chất lượng giấc ngủ.
            - **Phân tích đa biến:** Ma trận hệ số tương quan (Correlation Matrix Heatmap) giữa các biến số định lượng.
            """
        )

    with col_b:
        st.subheader("4.2. Mô Hình Dự Báo")
        st.markdown(
            """
            - **Hồi quy tuyến tính đơn biến:** Dự báo *Chất lượng giấc ngủ* theo *Thời lượng ngủ*.
            - **Hồi quy tuyến tính đa biến:** Dự báo *Chất lượng giấc ngủ* theo tổ hợp: *Thời lượng ngủ, Stress, Hoạt động thể chất, Huyết áp*.
            - **Hồi quy Logistics:** Phân loại nguy cơ mắc *Rối loạn giấc ngủ* theo BMI và Stress.
            """
        )

    with col_c:
        st.subheader("4.3. Chỉ Số Đánh Giá")
        st.markdown(
            """
            - **Đánh giá Hồi quy:**
              - $R^2$ (Hệ số xác định)
              - RMSE (Root Mean Squared Error)
              - MAE (Mean Absolute Error)
            - **Đánh giá Phân loại:**
              - Confusion Matrix
              - Accuracy, Precision, Recall, F1-Score
            """
        )