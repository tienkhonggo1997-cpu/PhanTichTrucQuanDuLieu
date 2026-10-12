import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Cấu hình giao diện Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def render_explorer(df):
    st.title("🔍 Chương 5: Phân Tích Thăm Dò Dữ Liệu (EDA)")
    st.caption(
        "Vận dụng phương pháp Thống kê Mô tả, kiểm tra Độ lệch (Skewness), Độ nhọn (Kurtosis) và Trực quan hóa dữ liệu."
    )
    st.markdown("---")

    # ==========================================================================
    # CHART 5.1: Boxplot Thời lượng ngủ theo Nghề nghiệp (Thống kê Mô tả)
    # ==========================================================================
    st.subheader("5.1 Thời Lượng Ngủ Theo Nghề Nghiệp (Thống kê Mô tả & Boxplot)")

    # 1. Hiển thị kết quả tính toán Thống kê mô tả
    st.markdown("##### 📋 Chỉ số Thống kê Mô tả của Thời lượng ngủ (Sleep Duration) theo Nghề nghiệp:")
    occ_stats = (
        df.groupby("Occupation")["Sleep Duration"]
        .agg(
            Số_người="count",
            Mean="mean",
            Std="std",
            Min="min",
            Median="median",
            Max="max",
        )
        .reset_index()
    )
    st.dataframe(
        occ_stats.style.format(
            {"Mean": "{:.2f}", "Std": "{:.2f}", "Min": "{:.1f}", "Median": "{:.1f}", "Max": "{:.1f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # 2. Vẽ Biểu đồ
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df, x="Occupation", y="Sleep Duration", palette="Set3", ax=ax1
    )
    plt.xticks(rotation=45)
    ax1.set_title(
        "Phân bố Thời lượng ngủ theo Lĩnh vực Nghề nghiệp",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Nghề nghiệp")
    ax1.set_ylabel("Thời lượng ngủ (Giờ)")
    plt.tight_layout()
    st.pyplot(fig1)

    # 3. Giải thích dựa trên chỉ số
    with st.expander("📌 **Giải thích & Phân tích dựa trên chỉ số (5.1)**", expanded=True):
        st.markdown(
            f"""
- **Trung vị (Median) & Trung bình (Mean) cao nhất:** Kỹ sư (*Engineer*) đạt **Median = 8.30h** (Mean = {df[df['Occupation']=='Engineer']['Sleep Duration'].mean():.2f}h), ngủ nhiều nhất và ổn định nhất.
- **Độ phân tán (Std - Độ lệch chuẩn):** Bác sĩ (*Doctor*, Std = {df[df['Occupation']=='Doctor']['Sleep Duration'].std():.2f}h) và Y tá (*Nurse*, Std = {df[df['Occupation']=='Nurse']['Sleep Duration'].std():.2f}h) có khoảng biến động giữa Min và Max rộng nhất ({df[df['Occupation']=='Doctor']['Sleep Duration'].min()}h - {df[df['Occupation']=='Doctor']['Sleep Duration'].max()}h).
- **Mức ngủ thấp nhất:** Nhân viên bán hàng (*Salesperson*) có **Median = 6.40h**, dao động từ Min = 6.00h đến Max = 6.50h.
        """
        )

    st.markdown("---")

    # ==========================================================================
    # CHART 5.2: Histogram & KDE Nhịp tim (Skewness & Kurtosis)
    # ==========================================================================
    st.subheader("5.2 Kiểm Tra Hình Dáng Phân Bố Nhịp Tim (Skewness & Kurtosis)")

    # 1. Tính toán Skewness, Kurtosis và các chỉ số hướng trung tâm
    hr_mean = df["Heart Rate"].mean()
    hr_median = df["Heart Rate"].median()
    hr_std = df["Heart Rate"].std()
    hr_skew = df["Heart Rate"].skew()
    hr_kurt = df["Heart Rate"].kurt()

    st.markdown("##### 📋 Kết quả tính toán Hình dáng phân bố (Shape Metrics) - Cột `Heart Rate`:")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean (Trung bình)", f"{hr_mean:.2f} BPM")
    m2.metric("Median (Trung vị)", f"{hr_median:.1f} BPM")
    m3.metric("Std (Độ lệch chuẩn)", f"{hr_std:.2f}")
    m4.metric("Skewness (Độ lệch)", f"{hr_skew:.2f}")
    m5.metric("Kurtosis (Độ nhọn)", f"{hr_kurt:.2f}")

    # 2. Vẽ Biểu đồ
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Heart Rate"], kde=True, color="teal", bins=15, ax=ax2)
    ax2.axvline(hr_mean, color="red", linestyle="--", label=f"Mean: {hr_mean:.1f}")
    ax2.axvline(hr_median, color="green", linestyle="-.", label=f"Median: {hr_median:.1f}")
    ax2.set_title(
        "Biểu đồ Phân bố Mật độ Nhịp tim (Heart Rate Distribution)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Nhịp tim (BPM)")
    ax2.set_ylabel("Tần số")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # 3. Giải thích dựa trên Skewness & Kurtosis
    with st.expander("📌 **Giải thích & Phân tích dựa trên Skewness & Kurtosis (5.2)**", expanded=True):
        st.markdown(
            f"""
- **Độ lệch (Skewness = {hr_skew:.2f} > 0):** Dữ liệu bị **lệch phải (Positive/Right Skewed)**. Điều này thể hiện qua việc **Mean ({hr_mean:.2f}) > Median ({hr_median:.1f})** do bị kéo về phía các giá trị nhịp tim cao.
- **Độ nhọn (Kurtosis = {hr_kurt:.2f} > 0):** Phân bố thuộc dạng **đỉnh nhọn (Leptokurtic)** hơn so với phân bố chuẩn, tập trung mật độ rất cao tại khoảng 68 - 72 BPM.
- **Hiện tượng Đa đỉnh (Multimodal):** Xuất hiện đỉnh phụ ở mức 75 - 80 BPM đại diện cho nhóm cá nhân có mức độ Stress cao hoặc gặp vấn đề về giấc ngủ.
        """
        )

    st.markdown("---")

    # ==========================================================================
    # CHART 5.3: Grouped Barplot BMI vs Sleep Disorder
    # ==========================================================================
    st.subheader("5.3 Rối Loạn Giấc Ngủ Theo Nhóm Thể Trạng BMI")

    # 1. Bảng thống kê tần suất
    st.markdown("##### 📋 Tần suất Rối loạn giấc ngủ theo Phân loại BMI:")
    bmi_ct = pd.crosstab(df["BMI Category"], df["Sleep Disorder"], margins=True, margins_name="Tổng")
    st.dataframe(bmi_ct, use_container_width=True)

    # 2. Vẽ Biểu đồ
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    sns.countplot(
        data=df, x="BMI Category", hue="Sleep Disorder", palette="Set2", ax=ax3
    )
    ax3.set_title(
        "Tỷ lệ Rối loạn Giấc ngủ theo Phân loại Thể trạng BMI",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xlabel("Phân loại BMI")
    ax3.set_ylabel("Số người")
    ax3.legend(title="Rối loạn giấc ngủ")
    plt.tight_layout()
    st.pyplot(fig3)

    # 3. Giải thích
    with st.expander("📌 **Giải thích & Phân tích dựa trên bảng số liệu (5.3)**", expanded=True):
        st.markdown(
            """
- **Nhóm Normal (Bình thường):** Chiếm đa số với **200/216 người (~92.6%)** không mắc rối loạn giấc ngủ (*None*).
- **Nhóm Overweight (Thừa cân):** Tỷ lệ mắc rối loạn tăng vọt với **129/148 người (~87.2%)**; trong đó Mất ngủ (*Insomnia*: 64 người) và Ngưng thở khi ngủ (*Sleep Apnea*: 65 người) chiếm tỷ lệ tương đương.
- **Nhóm Obese (Béo phì):** **100% (10/10 người)** đều ghi nhận có bệnh lý rối loạn giấc ngủ (6 *Sleep Apnea*, 4 *Insomnia*).
        """
        )

    st.markdown("---")

    # ==========================================================================
    # CHART 5.4: Scatterplot Stress vs Sleep Quality
    # ==========================================================================
    st.subheader("5.4 Tương Quan Căng Thẳng & Chất Lượng Ngủ (Thống kê Tương quan)")

    # 1. Tính trung bình Chất lượng ngủ theo Mức Stress
    stress_stats = (
        df.groupby("Stress Level")["Quality of Sleep"]
        .agg(Mean_Quality="mean", Min_Quality="min", Max_Quality="max", Count="count")
        .reset_index()
    )
    st.markdown("##### 📋 Thống kê Chất lượng giấc ngủ trung bình theo Mức độ Căng thẳng (Stress Level):")
    st.dataframe(
        stress_stats.style.format({"Mean_Quality": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # 2. Vẽ Biểu đồ
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x="Stress Level",
        y="Quality of Sleep",
        hue="Sleep Disorder",
        style="Gender",
        s=100,
        palette="bright",
        ax=ax4,
    )
    ax4.set_title(
        "Tương quan Căng thẳng & Chất lượng ngủ theo Rối loạn và Giới tính",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_xlabel("Mức độ Căng thẳng (Stress Level)")
    ax4.set_ylabel("Chất lượng Giấc ngủ (Quality of Sleep)")
    plt.tight_layout()
    st.pyplot(fig4)

    # 3. Giải thích
    with st.expander("📌 **Giải thích & Phân tích dựa trên số liệu (5.4)**", expanded=True):
        st.markdown(
            f"""
- **Tương quan nghịch rõ rệt:** Khi Stress Level = 3, Chất lượng ngủ đạt trung bình **{stress_stats[stress_stats['Stress Level']==3]['Mean_Quality'].values[0]:.2f}/10**. Ngược lại, khi Stress Level = 8, Chất lượng ngủ giảm mạnh xuống còn **{stress_stats[stress_stats['Stress Level']==8]['Mean_Quality'].values[0]:.2f}/10**.
- **Phân bố Bệnh lý:** *Insomnia* xuất hiện nhiều nhất ở các mức Stress trung bình-cao (mức 4 và 7), trong khi *Sleep Apnea* phân bố ở cả mức Stress thấp và rất cao.
        """
        )

    st.markdown("---")

    # ==========================================================================
    # CHART 5.5: Boxplot Số bước chân theo Trạng thái giấc ngủ (Thống kê Mô tả)
    # ==========================================================================
    st.subheader("5.5 Vận Động Hàng Ngày (Daily Steps) Theo Rối Loạn Giấc Ngủ")

    # 1. Hiển thị thống kê mô tả
    st.markdown("##### 📋 Thống kê Số bước chân hàng ngày (Daily Steps) theo Trạng thái Giấc ngủ:")
    steps_stats = (
        df.groupby("Sleep Disorder")["Daily Steps"]
        .agg(Mean="mean", Std="std", Min="min", Median="median", Max="max")
        .reset_index()
    )
    st.dataframe(
        steps_stats.style.format(
            {"Mean": "{:,.0f}", "Std": "{:,.0f}", "Min": "{:,.0f}", "Median": "{:,.0f}", "Max": "{:,.0f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # 2. Vẽ Biểu đồ
    fig5, ax5 = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df, x="Sleep Disorder", y="Daily Steps", palette="Pastel1", ax=ax5
    )
    ax5.set_title(
        "Phân bố Số bước chân Hàng ngày theo Trạng thái Giấc ngủ",
        fontsize=12,
        fontweight="bold",
    )
    ax5.set_xlabel("Trạng thái Giấc ngủ")
    ax5.set_ylabel("Số bước chân hàng ngày (Daily Steps)")
    plt.tight_layout()
    st.pyplot(fig5)

    # 3. Giải thích
    with st.expander("📌 **Giải thích & Phân tích (5.5)**", expanded=True):
        st.markdown(
            f"""
- **Nhóm Không mắc bệnh (None):** Duy trì vận động trung bình **{steps_stats[steps_stats['Sleep Disorder']=='None']['Mean'].values[0]:,.0f} bước/ngày** (Median = {steps_stats[steps_stats['Sleep Disorder']=='None']['Median'].values[0]:,.0f}).
- **Nhóm Insomnia (Mất ngủ):** Có mức vận động trung bình thấp hơn (**{steps_stats[steps_stats['Sleep Disorder']=='Insomnia']['Mean'].values[0]:,.0f} bước/ngày**).
- **Nhóm Sleep Apnea:** Mức độ biến động rất rộng (Min = {steps_stats[steps_stats['Sleep Disorder']=='Sleep Apnea']['Min'].values[0]:,.0f} đến Max = {steps_stats[steps_stats['Sleep Disorder']=='Sleep Apnea']['Max'].values[0]:,.0f} bước) do sự phân hóa giữa nhóm béo phì ít vận động và nhóm y tá đi lại nhiều.
        """
        )