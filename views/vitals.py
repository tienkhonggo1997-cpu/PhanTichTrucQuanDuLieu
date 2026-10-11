import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from pandas.plotting import parallel_coordinates

# Cấu hình giao diện Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def render_vitals(df):
    st.title("🛠️ Chương 4: Công Cụ Trực Quan Hóa Dữ Liệu")
    st.caption(
        "Ứng dụng các dạng biểu đồ chuyên sâu trong phân tích sinh hiệu và lối sống."
    )
    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 4.1: Area Plot
    # --------------------------------------------------------------------------
    st.subheader("4.1 Biểu Đồ Diện Tích Tích Lũy (Area Plot)")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    area_data = pd.crosstab(df["Age_Group"], df["Sleep Disorder"])
    area_data.plot(
        kind="area", stacked=True, alpha=0.7, colormap="Set2", ax=ax1
    )
    ax1.set_title(
        "Tích lũy Rối loạn Giấc ngủ theo Nhóm tuổi",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Nhóm tuổi")
    ax1.set_ylabel("Số lượng người")
    plt.tight_layout()
    st.pyplot(fig1)

    with st.expander("📌 **Giải thích & Phân tích (4.1)**", expanded=True):
        st.markdown(
            """
        - **Nhóm tuổi 20-35:** Tỷ lệ người không mắc rối loạn giấc ngủ (*None*) chiếm đa số.
        - **Nhóm tuổi 36-50 & 51-65:** Tỷ lệ Insomnia và Sleep Apnea tăng mạnh, mở rộng diện tích tích lũy.
        - **Kết luận:** Nguy cơ mắc rối loạn giấc ngủ gia tăng rõ rệt theo độ tuổi.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 4.2: Radar Chart
    # --------------------------------------------------------------------------
    st.subheader("4.2 Biểu Đồ Radar (Radar Chart)")

    categories = [
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
    ]
    top_occupations = df["Occupation"].value_counts().head(4).index

    df_radar = df.groupby("Occupation")[categories].mean()
    df_radar_norm = (df_radar - df_radar.min()) / (
        df_radar.max() - df_radar.min()
    )

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for occ in top_occupations:
        if occ in df_radar_norm.index:
            values = df_radar_norm.loc[occ].values.flatten().tolist()
            values += values[:1]
            ax2.plot(angles, values, linewidth=2, linestyle="solid", label=occ)
            ax2.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], categories)
    ax2.set_title(
        "So sánh Hồ sơ Chỉ số Sức khỏe theo Nghề nghiệp",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    st.pyplot(fig2)

    with st.expander("📌 **Giải thích & Phân tích (4.2)**", expanded=True):
        st.markdown(
            """
        - **Y tá (Nurse):** Mức độ hoạt động thể chất cao nhất trong bốn nghề (78.6 phút/ngày).
        - **Bác sĩ (Doctor):** Mức stress cao nhất (6.73) và chất lượng ngủ thấp nhất (6.65) trong bốn nghề.
        - **Kỹ sư (Engineer):** Thời lượng ngủ (7.99 giờ) và chất lượng ngủ (8.41) cao nhất, stress thấp nhất (3.89).
        - **Lưu ý:** Các trục được chuẩn hóa 0 - 1 theo trung bình của từng nghề, nên biểu đồ thể hiện thứ hạng tương đối giữa các nghề.
        - **Kết luận:** Phác họa bức chân dung đa chiều về rủi ro sức khỏe đặc thù theo từng ngành nghề.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 4.3: Heatmap
    # --------------------------------------------------------------------------
    st.subheader("4.3 Ma Trận Hệ Số Tương Quan (Heatmap)")

    num_cols = [
        col
        for col in [
            "Age",
            "Sleep Duration",
            "Quality of Sleep",
            "Physical Activity Level",
            "Stress Level",
            "Heart Rate",
            "Daily Steps",
            "Systolic_BP",
            "Diastolic_BP",
        ]
        if col in df.columns
    ]

    fig3, ax3 = plt.subplots(figsize=(9, 6))
    corr = df[num_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax3,
    )
    ax3.set_title(
        "Ma trận Hệ số Tương quan giữa các Biến số",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    st.pyplot(fig3)

    with st.expander("📌 **Giải thích & Phân tích (4.3)**", expanded=True):
        st.markdown(
            """
        - **Tương quan thuận mạnh:** *Sleep Duration* & *Quality of Sleep* ($r \\approx 0.88$).
        - **Tương quan nghịch mạnh:** *Stress Level* & *Quality of Sleep* ($r \\approx -0.90$).
        - **Kết luận:** Căng thẳng tâm lý là biến số tiêu cực tác động mạnh nhất đến chất lượng giấc ngủ.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 4.4: Jointplot / Marginal Plot
    # --------------------------------------------------------------------------
    st.subheader(
        "4.4 Mối Quan Hệ Thời Lượng & Chất Lượng Ngủ (Jointplot/Scatter)"
    )

    fig4 = px.scatter(
        df,
        x="Sleep Duration",
        y="Quality of Sleep",
        color="Sleep Disorder",
        marginal_x="histogram",
        marginal_y="histogram",
        title="Mối quan hệ giữa Thời lượng ngủ và Chất lượng ngủ",
    )
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("📌 **Giải thích & Phân tích (4.4)**", expanded=True):
        st.markdown(
            """
        - **Phân bố điểm:** Dải dữ liệu chéo đi lên thể hiện mối quan hệ tuyến tính đồng biến.
        - **Nhóm ngủ 7.5 - 8.5 giờ:** Ghi nhận chất lượng ngủ cao nhất (8-9 điểm) và đa số không bị rối loạn (*None*).
        - **Kết luận:** Ngủ đủ thời gian là điều kiện tiên quyết để đạt chất lượng giấc ngủ cao.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 4.5: Parallel Coordinates
    # --------------------------------------------------------------------------
    st.subheader("4.5 Biểu Đồ Tọa Độ Song Song (Parallel Coordinates)")

    cols_parallel = [
        "Sleep Duration",
        "Quality of Sleep",
        "Stress Level",
        "Heart Rate",
        "Sleep Disorder",
    ]
    df_p = df[cols_parallel].copy()
    for col in [
        "Sleep Duration",
        "Quality of Sleep",
        "Stress Level",
        "Heart Rate",
    ]:
        df_p[col] = (df_p[col] - df_p[col].min()) / (
            df_p[col].max() - df_p[col].min()
        )

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    parallel_coordinates(df_p, "Sleep Disorder", colormap="Set1", alpha=0.5, ax=ax5)
    ax5.set_title(
        "Biểu đồ Tọa độ Song song mô tả Đặc trưng Rối loạn Giấc ngủ",
        fontsize=12,
        fontweight="bold",
    )
    ax5.set_ylabel("Giá trị đã chuẩn hóa (0 - 1)")
    plt.tight_layout()
    st.pyplot(fig5)

    with st.expander("📌 **Giải thích & Phân tích (4.5)**", expanded=True):
        st.markdown(
            """
        - **Nhóm None:** Các đường nối tập trung ở *Sleep Duration* cao và *Stress Level* thấp.
        - **Nhóm Insomnia:** Đường nối gãy gập xuống ở *Sleep Duration* và vọt lên ở *Stress Level*.
        - **Kết luận:** Xác minh mẫu hình sinh hiệu đặc trưng phân tách các trạng thái bệnh lý.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 4.6: Pairplot
    # --------------------------------------------------------------------------
    st.subheader("4.6 Phân Mảnh Tương Quan Đa Biến (Pairplot)")

    pair_cols = [
        "Age",
        "Sleep Duration",
        "Physical Activity Level",
        "Stress Level",
        "Gender",
    ]
    pp = sns.pairplot(df[pair_cols], hue="Gender", palette="Set2", corner=True)
    pp.fig.suptitle(
        "Phân mảnh Tương quan Đa biến theo Giới tính",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    st.pyplot(pp.fig)

    with st.expander("📌 **Giải thích & Phân tích (4.6)**", expanded=True):
        st.markdown(
            """
        - **Phân bố theo giới tính:** Cho thấy sự khác biệt về mức độ vận động và độ tuổi khảo sát giữa Nam và Nữ.
        - **Xu hướng đa biến:** Giúp quan sát đồng thời sự phân cụm giữa thể chất, độ tuổi và mức độ căng thẳng.
        - **Kết luận:** Các chỉ số thể chất có khác biệt theo giới tính, nhưng giới tính gắn chặt với nghề nghiệp trong dữ liệu này nên chưa thể kết luận đó là ảnh hưởng của riêng giới tính.
        """
        )