import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_sleep_analysis(df):
    st.title("📊 Chương 3: Trực Quan Hóa & Phân Tích Dữ Liệu")
    st.caption(
        "Trực quan hóa dữ liệu đa chiều kết hợp phân tích chuyên sâu về các yếu tố ảnh hưởng đến Giấc ngủ và Sức khỏe."
    )
    st.markdown("---")

    # ====================================================
    # 1. BIỂU ĐỒ 1: AREA PLOT (Tích lũy Rối loạn Giấc ngủ theo Độ tuổi)
    # ====================================================
    st.subheader("1. Area Plot: Tích Lũy Rối Loạn Giấc Ngủ Theo Nhóm Tuổi")

    # Xử lý dữ liệu cho Area Plot
    if "Age_Group" not in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[20, 35, 50, 65],
            labels=["20-35 (Trẻ)", "36-50 (Trung niên)", "51-65 (Lớn tuổi)"],
        )

    age_pivot = df.pivot_table(
        index="Age_Group",
        columns="Sleep Disorder",
        values="Person ID",
        aggfunc="count",
        observed=False,
    ).fillna(0)

    fig1 = px.area(
        age_pivot,
        labels={"value": "Số lượng người", "Age_Group": "Nhóm tuổi"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Tỷ lệ tích lũy các dạng rối loạn giấc ngủ qua các nhóm tuổi",
    )
    fig1.update_layout(
        hovermode="x unified", legend_title_text="Rối loạn giấc ngủ"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Ô Nhận xét & Phân tích cho Biểu đồ 1
    with st.expander("📝 **Xem Nhận xét & Phân tích chi tiết (Chart 1)**", expanded=True):
        st.markdown(
            """
        * **Đánh giá chung:** 
            * Nhóm tuổi trẻ (**20-35**) hầu như có tỷ lệ giấc ngủ bình thường (*None*) rất cao và ít gặp các vấn đề nghiêm trọng về rối loạn giấc ngủ.
            * Khi độ tuổi tăng lên nhóm **36-50** và **51-65**, số lượng người mắc các chứng rối loạn giấc ngủ như **Insomnia (Mất ngủ)** và **Sleep Apnea (Ngưng thở khi ngủ)** tăng lên rõ rệt.
        * **Hàm ý / Đề xuất:** 
            * Tuổi tác là một yếu tố nguy cơ đáng kể đối với chất lượng giấc ngủ, cần có các biện pháp can thiệp và chăm sóc sức khỏe giấc ngủ sớm cho nhóm tuổi trung niên và người lớn tuổi.
        """
        )

    st.markdown("---")

    # ====================================================
    # 2. BIỂU ĐỒ 2: SCATTER PLOT (Huyết Áp Tâm Thu vs Tâm Trương)
    # ====================================================
    st.subheader("2. Scatter Plot: Tương Quan Huyết Áp Tâm Thu & Tâm Trương")

    fig2 = px.scatter(
        df,
        x="Systolic_BP",
        y="Diastolic_BP",
        color="Sleep Disorder",
        size="Heart Rate",
        hover_data=["Age", "Occupation", "BMI Category"],
        labels={
            "Systolic_BP": "Huyết áp Tâm Thu (Systolic BP)",
            "Diastolic_BP": "Huyết áp Tâm Trương (Diastolic BP)",
        },
        title="Phân bố chỉ số Huyết áp phân loại theo Rối loạn giấc ngủ",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Ô Nhận xét & Phân tích cho Biểu đồ 2
    with st.expander("📝 **Xem Nhận xét & Phân tích chi tiết (Chart 2)**", expanded=True):
        st.markdown(
            """
        * **Đánh giá chung:**
            * Có sự phân cụm rõ ràng: Những người thuộc nhóm **Sleep Apnea (Ngưng thở khi ngủ)** có xu hướng tập trung ở khu vực có chỉ số **Huyết áp Tâm thu (>130 mmHg)** và **Tâm trương (>85 mmHg)** cao hơn mức bình thường.
            * Kích thước bong bóng (Nhịp tim - Heart Rate) cũng có xu hướng lớn hơn ở nhóm bị ngưng thở khi ngủ.
        * **Hàm ý / Đề xuất:**
            * Rối loạn giấc ngủ dạng ngưng thở có mối liên hệ chặt chẽ với nguy cơ tăng huyết áp và các bệnh lý tim mạch.
        """
        )

    st.markdown("---")

    # ====================================================
    # 3. BIỂU ĐỒ 3: BOX PLOT (Thời Gian Ngủ Theo Nghề Nghiệp)
    # ====================================================
    st.subheader("3. Box Plot: Phân Bố Thời Gian Ngủ Theo Nghề Nghiệp")

    fig3 = px.box(
        df,
        x="Occupation",
        y="Sleep Duration",
        color="Gender",
        labels={
            "Occupation": "Nghề nghiệp",
            "Sleep Duration": "Thời gian ngủ (Giờ/Ngày)",
        },
        title="Biến thiên thời gian ngủ trung bình giữa các ngành nghề",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig3.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig3, use_container_width=True)

    # Ô Nhận xét & Phân tích cho Biểu đồ 3
    with st.expander("📝 **Xem Nhận xét & Phân tích chi tiết (Chart 3)**", expanded=True):
        st.markdown(
            """
        * **Đánh giá chung:**
            * **Bác sĩ (Doctor), Kỹ sư (Engineer), Bác sĩ thú y (Veterinarian)** có thời gian ngủ tương đối ổn định và thời lượng ngủ trung bình cao (khoảng 7 - 8 giờ/ngày).
            * **Y tá (Nurse) và Đại diện bán hàng (Sales Representative)** ghi nhận thời lượng ngủ ngắn hơn và độ biến thiên (chênh lệch) giữa các cá nhân rất cao.
        * **Hàm ý / Đề xuất:**
            * Đặc thù công việc (như trực ca đêm của Y tá hay áp lực chỉ tiêu của Dân bán hàng) ảnh hưởng trực tiếp và nghiêm trọng đến quỹ thời gian nghỉ ngơi hàng ngày.
        """
        )

    st.markdown("---")

    # ====================================================
    # 4. BIỂU ĐỒ 4: HEATMAP (Mức Độ Stress vs Chất Lượng Giấc Ngủ)
    # ====================================================
    st.subheader("4. Heatmap: Ma Trận Tương Quan Giữa Stress & Chất Lượng Giấc Ngủ")

    stress_quality_pivot = df.pivot_table(
        index="Stress Level",
        columns="Quality of Sleep",
        values="Person ID",
        aggfunc="count",
    ).fillna(0)

    fig4 = px.imshow(
        stress_quality_pivot,
        labels=dict(
            x="Chất Lượng Giấc Ngủ (1-10)",
            y="Mức Độ Stress (1-10)",
            color="Số lượng người",
        ),
        x=stress_quality_pivot.columns,
        y=stress_quality_pivot.index,
        color_continuous_scale="Viridis",
        title="Mật độ phân bố giữa Mức độ Căng thẳng và Chất lượng Giấc ngủ",
        text_auto=True,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Ô Nhận xét & Phân tích cho Biểu đồ 4
    with st.expander("📝 **Xem Nhận xét & Phân tích chi tiết (Chart 4)**", expanded=True):
        st.markdown(
            """
        * **Đánh giá chung:**
            * Mối tương quan nghịch rõ rệt: Những người báo cáo **Mức độ Stress cao (7-8/10)** tập trung chủ yếu ở mức **Chất lượng giấc ngủ thấp (4-6/10)**.
            * Ngược lại, nhóm có **Stress thấp (3-4/10)** sở hữu **Chất lượng giấc ngủ rất cao (8-9/10)**.
        * **Hàm ý / Đề xuất:**
            * Quản lý căng thẳng (Stress management) là chìa khóa cốt lõi để cải thiện trực tiếp chất lượng giấc ngủ cho đối tượng khảo sát.
        """
        )