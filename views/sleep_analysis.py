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
            * **Insomnia (Mất ngủ)** tập trung ở nhóm **36-50** (65 người), sau đó giảm mạnh ở nhóm **51-65** (7 người).
            * **Sleep Apnea (Ngưng thở khi ngủ)** tăng dần theo tuổi và chiếm tỷ lệ cao nhất ở nhóm **51-65** (33/76 người).
        * **Hàm ý / Đề xuất:**
            * Tuổi tác có liên hệ với dạng rối loạn giấc ngủ: nhóm trung niên cần chú ý đến mất ngủ, nhóm lớn tuổi cần chú ý đến ngưng thở khi ngủ.
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
            * **Kỹ sư (Engineer)** có thời lượng ngủ cao và ổn định nhất (trung vị 8.3 giờ/ngày); **Nhân viên bán hàng (Salesperson)** thấp nhất trong các nghề đông người (trung vị 6.4 giờ/ngày).
            * **Bác sĩ (Doctor) và Y tá (Nurse)** có độ biến thiên giữa các cá nhân rộng nhất (khoảng tứ phân vị lần lượt 6.05 - 7.7 và 6.1 - 8.1 giờ).
            * Các nghề Manager, Sales Representative, Scientist, Software Engineer chỉ có 1 - 4 người nên không đủ để nhận xét.
        * **Hàm ý / Đề xuất:**
            * Thời lượng ngủ có liên hệ với nghề nghiệp. Tuy nhiên nghề nghiệp và giới tính trong dữ liệu gần như trùng nhau (Y tá hầu hết là nữ, Bác sĩ hầu hết là nam) nên không tách riêng được hai yếu tố này.
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
            * Mức độ căng thẳng có liên hệ nghịch rất mạnh với chất lượng giấc ngủ; quản lý căng thẳng (Stress management) là hướng đáng ưu tiên khi muốn cải thiện giấc ngủ.
        """
        )