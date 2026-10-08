import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
        - **Trung vị cao:** Kỹ sư (*Engineer*) và Kế toán (*Accountant*) đạt gần 8.0 giờ ngủ ổn định.
        - **Khoảng tứ phân vị rộng:** Bác sĩ (*Doctor*) và Y tá (*Nurse*) thể hiện sự biến động lớn do ca trực.
        - **Trung vị thấp nhất:** Nhân viên Kinh doanh (*Sales Representative*) ngủ dưới 6.0 giờ.
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
        - **Nhóm Normal:** Đại đa số thuộc nhóm *None* (không mắc bệnh lý giấc ngủ).
        - **Nhóm Overweight:** Tỷ lệ người bị Mất ngủ (*Insomnia*) chiếm ưu thế tuyệt đối.
        - **Nhóm Obese:** Tỷ lệ bị Ngưng thở khi ngủ (*Sleep Apnea*) chiếm đa số.
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
        - **Cụm Stress thấp (3-4) / Chất lượng ngủ cao (8-9):** Hoàn toàn là nhóm *None*.
        - **Cụm Stress cao (7-8) / Chất lượng ngủ thấp (4-5):** Bao gồm các trường hợp mắc bệnh lý.
        - **Kết luận:** Mức độ stress là ranh giới rõ ràng nhất phân tách giữa các nhóm bệnh lý.
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
        - **Nhóm None:** Duy trì vận động ở mức vừa phải (6,000 - 8,000 bước/ngày).
        - **Nhóm Sleep Apnea:** Xuất hiện phân hóa giữa nhóm béo phì ít vận động và nhóm Y tá đi lại nhiều.
        - **Kết luận:** Vận động giúp hỗ trợ sức khỏe nhưng không loại bỏ hoàn toàn nguy cơ do BMI và Stress.
        """
        )