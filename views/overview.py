import plotly.express as px
import streamlit as st


def render_overview(df):
    st.title("🚀 Chương 1: Tổng Quan & Thống Kê Nhanh (KPIs)")
    st.caption("Báo cáo tổng hợp các chỉ số trọng yếu về sức khỏe giấc ngủ.")
    st.markdown("---")

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tổng Số Người", f"{len(df)} người")
    k2.metric("Thời Gian Ngủ TB", f"{df['Sleep Duration'].mean():.1f}h")
    k3.metric("Chất Lượng Ngủ", f"{df['Quality of Sleep'].mean():.1f}/10")
    k4.metric("Stress Level", f"{df['Stress Level'].mean():.1f}/10")
    k5.metric("Bước Chân TB", f"{int(df['Daily Steps'].mean()):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Tỷ Lệ Rối Loạn Giấc Ngủ")
        fig = px.pie(df, names="Sleep Disorder", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⚖️ Phân Bố Theo Nhóm BMI")
        fig = px.histogram(df, x="BMI Category", color="Gender", barmode="group")
        st.plotly_chart(fig, use_container_width=True)