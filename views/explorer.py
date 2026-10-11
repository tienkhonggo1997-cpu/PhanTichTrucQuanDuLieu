import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

# Cấu hình giao diện Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def render_explorer(df):
    st.title("🔍 Chương 5: Phân Tích Thăm Dò Dữ Liệu (EDA)")
    st.caption(
        "Khám phá mối tương quan sâu hơn giữa môi trường làm việc, lối sống và rối loạn giấc ngủ."
    )
    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 5.1: Boxplot Thời lượng ngủ theo Nghề nghiệp
    # --------------------------------------------------------------------------
    st.subheader("5.1 Thời Lượng Ngủ Theo Nghề Nghiệp (Boxplot)")

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

    with st.expander("📌 **Giải thích & Phân tích (5.1)**", expanded=True):
        st.markdown(
            """
        - **Trung vị cao nhất:** Kỹ sư (*Engineer*) đạt 8.3 giờ; Bác sĩ (*Doctor*) 7.6 giờ, Luật sư (*Lawyer*) 7.3 giờ, Kế toán (*Accountant*) 7.2 giờ.
        - **Khoảng tứ phân vị rộng:** Bác sĩ (*Doctor*) và Y tá (*Nurse*) có mức biến động lớn nhất giữa các cá nhân.
        - **Trung vị thấp nhất (nghề từ 30 người trở lên):** Nhân viên bán hàng (*Salesperson*) 6.4 giờ. *Sales Representative* thấp hơn (5.9 giờ) nhưng chỉ có 2 người nên không đủ để kết luận.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 5.2: Histogram & KDE Nhịp tim
    # --------------------------------------------------------------------------
    st.subheader("5.2 Phân Bố Mật Độ Nhịp Tim (Histogram & KDE)")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Heart Rate"], kde=True, color="teal", bins=15, ax=ax2)
    ax2.set_title(
        "Biểu đồ Phân bố Mật độ Nhịp tim (Heart Rate Distribution)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("Nhịp tim (BPM)")
    ax2.set_ylabel("Tần số")
    plt.tight_layout()
    st.pyplot(fig2)

    with st.expander("📌 **Giải thích & Phân tích (5.2)**", expanded=True):
        st.markdown(
            """
        - **Đỉnh chính:** Tập trung tại khoảng 68 - 72 nhịp/phút ở đối tượng bình thường.
        - **Đỉnh phụ (Đa đỉnh):** Xuất hiện thêm cụm nhỏ ở 75 - 80 BPM đại diện nhóm có chỉ số stress cao.
        - **Kết luận:** Nhịp tim không phân bố chuẩn hoàn toàn mà chia theo trạng thái sức khỏe.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 5.3: Grouped Barplot BMI vs Sleep Disorder
    # --------------------------------------------------------------------------
    st.subheader("5.3 Rối Loạn Giấc Ngủ Theo Nhóm Thể Trạng BMI")

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

    with st.expander("📌 **Giải thích & Phân tích (5.3)**", expanded=True):
        st.markdown(
            """
        - **Nhóm Normal:** 200/216 người (93%) thuộc nhóm *None* (không mắc bệnh lý giấc ngủ).
        - **Nhóm Overweight:** 129/148 người (87%) mắc rối loạn; Mất ngủ (*Insomnia*, 64 người) và Ngưng thở khi ngủ (*Sleep Apnea*, 65 người) gần như ngang nhau.
        - **Nhóm Obese:** Cả 10 người đều mắc rối loạn (6 *Sleep Apnea*, 4 *Insomnia*); nhóm này quá nhỏ để so sánh tỷ lệ.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 5.4: Scatterplot Stress vs Sleep Quality
    # --------------------------------------------------------------------------
    st.subheader(
        "5.4 Tương Quan Căng Thẳng & Chất Lượng Ngủ Nâng Cao (Scatterplot)"
    )

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

    with st.expander("📌 **Giải thích & Phân tích (5.4)**", expanded=True):
        st.markdown(
            """
        - **Xu hướng chung:** Stress càng cao thì chất lượng ngủ càng thấp (trung bình từ 8.97 ở mức stress 3 xuống 5.86 ở mức stress 8).
        - **Theo bệnh lý:** *Insomnia* tập trung ở mức stress 4 và 7; *Sleep Apnea* xuất hiện ở cả hai đầu (30 người ở mức stress 3 và 30 người ở mức stress 8).
        - **Kết luận:** Stress liên hệ chặt với chất lượng ngủ, nhưng riêng mức stress không đủ để phân tách các nhóm bệnh lý.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 5.5: Boxplot Số bước chân theo Trạng thái giấc ngủ
    # --------------------------------------------------------------------------
    st.subheader("5.5 Vận Động Hàng Ngày (Daily Steps) Theo Rối Loạn Giấc Ngủ")

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

    with st.expander("📌 **Giải thích & Phân tích (5.5)**", expanded=True):
        st.markdown(
            """
        - **Nhóm None:** Duy trì vận động ở mức vừa phải (phần lớn trong khoảng 5,000 - 8,000 bước/ngày).
        - **Nhóm Sleep Apnea:** Xuất hiện phân hóa giữa nhóm béo phì ít vận động và nhóm Y tá đi lại nhiều.
        - **Kết luận:** Vận động giúp hỗ trợ sức khỏe nhưng không loại bỏ hoàn toàn nguy cơ do BMI và Stress.
        """
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Chart 5.6: Kiểm định ANOVA Thời lượng ngủ theo Rối loạn giấc ngủ
    # --------------------------------------------------------------------------
    st.subheader("5.6 Kiểm Định ANOVA: Thời Lượng Ngủ Theo Rối Loạn Giấc Ngủ")

    fig6, ax6 = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=df,
        x="Sleep Disorder",
        y="Sleep Duration",
        hue="Sleep Disorder",
        errorbar="sd",
        palette="viridis",
        legend=False,
        ax=ax6,
    )
    ax6.set_title(
        "Thời lượng ngủ trung bình (± độ lệch chuẩn) theo Trạng thái Giấc ngủ",
        fontsize=12,
        fontweight="bold",
    )
    ax6.set_xlabel("Trạng thái Giấc ngủ")
    ax6.set_ylabel("Thời lượng ngủ trung bình (Giờ)")
    plt.tight_layout()
    st.pyplot(fig6)

    # Kiểm định ANOVA một yếu tố: thời lượng ngủ có khác nhau giữa các nhóm không?
    anova_groups = {
        name: g["Sleep Duration"] for name, g in df.groupby("Sleep Disorder")
    }
    if len(anova_groups) >= 2 and all(len(g) >= 2 for g in anova_groups.values()):
        f_stat, p_value = stats.f_oneway(*anova_groups.values())

        col_f, col_p = st.columns(2)
        col_f.metric("Giá trị F", f"{f_stat:.2f}")
        col_p.metric("p-value", f"{p_value:.2e}")

        means_text = ", ".join(
            f"*{name}* {g.mean():.2f} giờ" for name, g in anova_groups.items()
        )
        if p_value < 0.05:
            conclusion = "thời lượng ngủ trung bình giữa các nhóm **khác nhau có ý nghĩa thống kê** ở mức 0.05"
        else:
            conclusion = "**chưa đủ bằng chứng** để nói thời lượng ngủ trung bình giữa các nhóm khác nhau ở mức 0.05"

        with st.expander("📌 **Giải thích & Phân tích (5.6)**", expanded=True):
            st.markdown(
                f"""
            - **Trung bình từng nhóm:** {means_text}.
            - **Kết quả kiểm định:** F = {f_stat:.2f}, p-value = {p_value:.2e}.
            - **Kết luận:** {conclusion}.
            """
            )
    else:
        st.info("Bộ lọc hiện tại không đủ nhóm để chạy kiểm định ANOVA.")
