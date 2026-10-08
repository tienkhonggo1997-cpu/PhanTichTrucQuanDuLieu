import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def render_evaluation(df):
    st.title("📊 Chương 7: Đánh Giá & Lựa Chọn Mô Hình (Model Evaluation & Selection)")
    st.markdown(
        "Phân tích hiệu năng mô hình, hiện tượng Overfitting/Underfitting và kỹ thuật Điều chuẩn (Regularization) Ridge.")

    # Kiểm tra biến mục tiêu và các biến độc lập
    Y = df['Quality of Sleep']
    X_data = df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity Level']]

    # Khởi tạo 3 Tabs dựa trên 3 phần phân tích của Chương 7
    tab1, tab2, tab3 = st.tabs([
        "🧪 Tab 1: Train/Test Split & Cross Validation",
        "⚖️ Tab 2: Phân tích Underfitting & Overfitting",
        "🎯 Tab 3: Hồi quy Ridge & Tối ưu Alpha"
    ])

    # =============================================================================
    # TAB 1: TRAIN/TEST SPLIT & CROSS VALIDATION
    # =============================================================================
    with tab1:
        st.header("1. Đánh giá Mô hình trên Tập Test & Cross-Validation")
        st.markdown(
            "Tách tập dữ liệu theo tỉ lệ **70% Train / 30% Test** và đánh giá độ ổn định qua **5-Fold Cross Validation**.")

        col_code1, col_res1 = st.columns([1, 1])

        # Thực thi tính toán
        x_train, x_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3, random_state=42)

        lr = LinearRegression()
        lr.fit(x_train, y_train)

        train_r2 = lr.score(x_train, y_train)
        test_r2 = lr.score(x_test, y_test)

        cv_scores = cross_val_score(lr, X_data, Y, cv=5)
        cv_mean_r2 = cv_scores.mean()

        with col_code1:
            st.subheader("💻 Code Python tính toán")
            code_tab1 = """
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

Y = df['Quality of Sleep']
X_data = df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity Level']]

# 1. Tách Train / Test (70% - 30%)
x_train, x_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

train_r2 = lr.score(x_train, y_train)
test_r2 = lr.score(x_test, y_test)

# 2. Cross Validation (5-Fold)
cv_scores = cross_val_score(lr, X_data, Y, cv=5)
cv_mean_r2 = cv_scores.mean()
"""
            st.code(code_tab1, language='python')

        with col_res1:
            st.subheader("📋 Kết quả Đánh giá")
            st.metric("Train R² Score", f"{train_r2:.4f}")
            st.metric("Test R² Score (30%)", f"{test_r2:.4f}")
            st.metric("5-Fold Cross-Validation R² Mean", f"{cv_mean_r2:.4f}")

            # Bảng hiển thị điểm số từng Fold
            cv_df = pd.DataFrame({
                'Fold': [f'Fold {i + 1}' for i in range(len(cv_scores))],
                'R² Score': cv_scores
            })
            st.dataframe(cv_df, hide_index=True, use_container_width=True)

        st.subheader("📉 Biểu đồ Phân phối (Distribution Plot) tập Test: Thực tế vs Dự đoán")

        y_test_pred = lr.predict(x_test)

        # Vẽ biểu đồ KDE/Distribution bằng Plotly
        fig1 = go.Figure()

        # Đường thực tế (Actual)
        fig1.add_trace(go.Histogram(
            x=y_test, histnorm='probability density', name='Giá trị Thực tế (Actual)',
            opacity=0.4, marker_color='red'
        ))

        # Đường dự đoán (Predicted)
        fig1.add_trace(go.Histogram(
            x=y_test_pred, histnorm='probability density', name='Dự đoán Tập Test (Predicted)',
            opacity=0.4, marker_color='blue'
        ))

        fig1.update_layout(
            title="1. Đánh giá Mô hình trên Tập Test (30%)",
            xaxis_title="Quality of Sleep",
            yaxis_title="Density",
            barmode='overlay'
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - **Khả năng tổng quát hóa (Generalization):** Sự chênh lệch giữa **Train $R^2$ ({train_r2:.4f})** và **Test $R^2$ ({test_r2:.4f})** rất nhỏ, cho thấy mô hình không bị overfitting quá mức khi kiểm thử dữ liệu mới.
        - **Độ ổn định (5-Fold CV):** Giá trị $R^2$ trung bình qua 5 lượt kiểm tra đạt **{cv_mean_r2:.4f}**, khẳng định tập dữ liệu có tính đồng nhất cao và mô hình hoạt động ổn định.
        """)

    # =============================================================================
    # TAB 2: PHÂN TÍCH UNDERFITTING & OVERFITTING
    # =============================================================================
    with tab2:
        st.header("2. Phân tích Underfitting & Overfitting qua Bậc Mô Hình")
        st.markdown(
            "Khảo sát sự thay đổi của chỉ số $R^2$ trên tập **Train** và **Test** khi tăng dần bậc đa thức (Degree).")

        col_code2, col_res2 = st.columns([1, 1])

        # Tính toán đa thức qua các bậc
        degrees = [1, 2, 3, 4, 5]
        r2_train, r2_test = [], []

        for d in degrees:
            poly = PolynomialFeatures(degree=d)
            x_tr_p = poly.fit_transform(x_train[['Stress Level']])
            x_te_p = poly.transform(x_test[['Stress Level']])

            m = LinearRegression().fit(x_tr_p, y_train)
            r2_train.append(m.score(x_tr_p, y_train))
            r2_test.append(m.score(x_te_p, y_test))

        with col_code2:
            st.subheader("💻 Code Python tính toán")
            code_tab2 = """
degrees = [1, 2, 3, 4, 5]
r2_train, r2_test = [], []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    x_tr_p = poly.fit_transform(x_train[['Stress Level']])
    x_te_p = poly.transform(x_test[['Stress Level']])

    m = LinearRegression().fit(x_tr_p, y_train)
    r2_train.append(m.score(x_tr_p, y_train))
    r2_test.append(m.score(x_te_p, y_test))
"""
            st.code(code_tab2, language='python')

        with col_res2:
            st.subheader("📋 Kết quả theo từng Bậc đa thức")
            degree_df = pd.DataFrame({
                'Bậc Đa Thức (Degree)': degrees,
                'Train R² Score': r2_train,
                'Test R² Score': r2_test
            })
            st.dataframe(degree_df, hide_index=True, use_container_width=True)

        st.subheader("📈 Biểu đồ Đường: Train R² vs Test R²")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=degrees, y=r2_train, mode='lines+markers', name='Train R² (Độ khớp tập Train)',
                                  line=dict(color='blue', width=2)))
        fig2.add_trace(go.Scatter(x=degrees, y=r2_test, mode='lines+markers', name='Test R² (Độ khớp tập Test)',
                                  line=dict(color='red', width=2)))

        fig2.update_layout(
            title="2. Chọn Bậc Mô hình: Phân tích Underfitting & Overfitting",
            xaxis_title="Bậc đa thức (Degree)",
            yaxis_title="Hệ số xác định R²"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown("""
        - **Underfitting (Bậc quá thấp):** Ở các bậc đa thức nhỏ, mô hình có thể quá đơn giản để bắt được xu hướng thực tế của dữ liệu.
        - **Overfitting (Bậc quá cao):** Khi bậc đa thức tăng lên quá cao, điểm $R^2$ trên tập Train tiếp tục tăng hoặc giữ ở mức cao nhưng $R^2$ trên tập Test giảm sút đáng kể. Điểm gãy giữa Train và Test chính là dấu hiệu của hiện tượng học vẹt (Overfitting).
        """)

    # =============================================================================
    # TAB 3: HỒI QUY RIDGE & TỐI ƯU ALPHA
    # =============================================================================
    with tab3:
        st.header("3. Hồi quy Ridge: Tối ưu hóa Tham số Điều chuẩn (Alpha)")
        st.markdown(
            "Ứng dụng thuật toán **Ridge Regression** kết hợp chuẩn hóa dữ liệu (`StandardScaler`) trên đa thức bậc 3 để kiểm soát Overfitting.")

        col_code3, col_res3 = st.columns([1, 1])

        # 1. Chuẩn hóa dữ liệu & Tạo biến đa thức bậc 3
        scaler = StandardScaler()
        x_tr_scaled = scaler.fit_transform(x_train)
        x_te_scaled = scaler.transform(x_test)

        poly3 = PolynomialFeatures(degree=3, include_bias=False)
        x_tr_p3 = poly3.fit_transform(x_tr_scaled)
        x_te_p3 = poly3.transform(x_te_scaled)

        # 2. Tạo dãy leo thang alpha (Log scale)
        alphas = np.logspace(-2, 4, 100)
        r_train_scores, r_test_scores = [], []

        for a in alphas:
            ridge_model = Ridge(alpha=a)
            ridge_model.fit(x_tr_p3, y_train)
            r_train_scores.append(ridge_model.score(x_tr_p3, y_train))
            r_test_scores.append(ridge_model.score(x_te_p3, y_test))

        # Tìm Alpha tối ưu
        best_idx = np.argmax(r_test_scores)
        best_alpha = alphas[best_idx]
        best_test_r2 = r_test_scores[best_idx]

        with col_code3:
            st.subheader("💻 Code Python tính toán")
            code_tab3 = """
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Chuẩn hóa & Biến đổi đa thức bậc 3
scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_train)
x_te_scaled = scaler.transform(x_test)

poly3 = PolynomialFeatures(degree=3, include_bias=False)
x_tr_p3 = poly3.fit_transform(x_tr_scaled)
x_te_p3 = poly3.transform(x_te_scaled)

# Quét giá trị Alpha
alphas = np.logspace(-2, 4, 100)
for a in alphas:
    ridge_model = Ridge(alpha=a).fit(x_tr_p3, y_train)
    r_train_scores.append(ridge_model.score(x_tr_p3, y_train))
    r_test_scores.append(ridge_model.score(x_te_p3, y_test))
"""
            st.code(code_tab3, language='python')

        with col_res3:
            st.subheader("📋 Kết quả Tối ưu Ridge")
            st.metric("Alpha tối ưu (Best Alpha)", f"{best_alpha:.4f}")
            st.metric("Test R² Đạt được", f"{best_test_r2:.4f}")
            st.metric("Train R² Tương ứng", f"{r_train_scores[best_idx]:.4f}")

        st.subheader("📊 Biểu đồ Mối liên quan giữa Alpha (Log Scale) và R²")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=alphas, y=r_train_scores, mode='lines', name='Train R²', line=dict(color='blue')))
        fig3.add_trace(go.Scatter(x=alphas, y=r_test_scores, mode='lines', name='Validation Test R²',
                                  line=dict(color='red', width=2)))

        # Vạch đường thẳng alpha tối ưu
        fig3.add_vline(x=best_alpha, line_dash="dash", line_color="green",
                       annotation_text=f"Alpha tối ưu ({best_alpha:.2f})")

        fig3.update_layout(
            title="3. Hồi quy Ridge: Mối liên quan giữa Alpha và R²",
            xaxis_title="Hệ số điều chuẩn Alpha (Log scale)",
            yaxis_title="Hệ số xác định R²",
            xaxis_type="log"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - **Vai trò của Alpha:** Khi $\\alpha$ quá nhỏ, mô hình giống như Hồi quy đa thức thông thường (dễ bị Overfitting). Khi $\\alpha$ quá lớn, mô hình bị phạt trọng số quá mạnh dẫn đến Underfitting.
        - **Giá trị tối ưu:** Với hệ số $\\alpha = {best_alpha:.4f}$, mô hình Ridge đạt được điểm $R^2$ kiểm thử cao nhất là **{best_test_r2:.4f}**, cân bằng tốt nhất giữa độ phức tạp và khả năng dự đoán.
        """)