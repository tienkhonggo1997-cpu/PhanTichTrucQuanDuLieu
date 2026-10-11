import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from modules.charts import kde_trace


def render_regression(df):
    st.title("📊 Chương 6: Mô Hình Hồi Quy (Model Development)")
    st.markdown("Xây dựng và đánh giá các mô hình hồi quy (SLR, MLR, Polynomial Regression, Pipeline).")
    st.info(
        "ℹ️ R² và MSE trong chương này được tính trên **chính dữ liệu dùng để huấn luyện** (in-sample), "
        "nên mô hình càng phức tạp thì R² càng cao. Khả năng dự đoán trên dữ liệu mới được đánh giá ở Chương 7."
    )

    # Tạo 4 Tab ứng với 4 phần trong code mẫu
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Tab 1: Hồi quy Đơn tuyến tính (SLR)",
        "📊 Tab 2: Hồi quy Đa biến (MLR)",
        "🌀 Tab 3: Hồi quy Đa thức (Polynomial)",
        "⚙️ Tab 4: Pipeline (StandardScaler + Poly + SLR)"
    ])

    # =============================================================================
    # TAB 1: SIMPLE LINEAR REGRESSION (SLR)
    # =============================================================================
    with tab1:
        st.header("1. Hồi quy Đơn tuyến tính (Simple Linear Regression)")
        st.markdown("Mối quan hệ giữa **Sleep Duration** và **Quality of Sleep**.")

        # Tính toán SLR
        X_slr = df[['Sleep Duration']]
        Y = df['Quality of Sleep']

        lm_slr = LinearRegression().fit(X_slr, Y)
        y_pred_slr = lm_slr.predict(X_slr)

        r2_slr = lm_slr.score(X_slr, Y)
        mse_slr = mean_squared_error(Y, y_pred_slr)

        col_code1, col_res1 = st.columns([1, 1])

        with col_code1:
            st.subheader("💻 Code Python tính toán")
            code_tab1 = """
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_slr = df[['Sleep Duration']]
Y = df['Quality of Sleep']

lm_slr = LinearRegression().fit(X_slr, Y)
y_pred_slr = lm_slr.predict(X_slr)

print(f'SLR Equation: Quality of Sleep = {lm_slr.intercept_:.4f} + {lm_slr.coef_[0]:.4f} * Sleep Duration')
print(f'R2 score: {lm_slr.score(X_slr, Y):.4f}')
print(f'MSE: {mean_squared_error(Y, y_pred_slr):.4f}')
"""
            st.code(code_tab1, language='python')

        with col_res1:
            st.subheader("📋 Kết quả tính toán")
            st.metric("Phương trình SLR", f"Quality of Sleep = {lm_slr.intercept_:.4f} + {lm_slr.coef_[0]:.4f} * Sleep Duration")
            st.metric("R² Score", f"{r2_slr:.4f}")
            st.metric("MSE", f"{mse_slr:.4f}")

        # Biểu đồ
        col_fig1, col_fig2 = st.columns(2)

        with col_fig1:
            st.subheader("1.1. Regression Plot")
            fig1_1 = px.scatter(
                df, x='Sleep Duration', y='Quality of Sleep',
                title="1.1. Sleep Duration vs Quality of Sleep",
                opacity=0.5, color_discrete_sequence=['blue']
            )
            # Thêm đường trendline
            x_range = np.linspace(df['Sleep Duration'].min(), df['Sleep Duration'].max(), 100).reshape(-1, 1)
            y_range_pred = lm_slr.predict(x_range)
            fig1_1.add_trace(go.Scatter(x=x_range.flatten(), y=y_range_pred, mode='lines', name='Regression Line', line=dict(color='blue', width=2)))
            st.plotly_chart(fig1_1, use_container_width=True)

        with col_fig2:
            st.subheader("1.2. Residual Plot")
            residuals_slr = Y - y_pred_slr
            fig1_2 = px.scatter(
                x=df['Sleep Duration'], y=residuals_slr,
                title="1.2. Residual Plot: Phần dư mô hình SLR",
                labels={'x': 'Sleep Duration', 'y': 'Residuals'},
                color_discrete_sequence=['purple'], opacity=0.7
            )
            fig1_2.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig1_2, use_container_width=True)

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - **Chỉ số R² ({r2_slr:.4f}):** Thời lượng ngủ giải thích khoảng **{r2_slr*100:.2f}%** sự biến thiên của chất lượng giấc ngủ.
        - **Phần dư (Residual Plot):** Các điểm phân bố ngẫu nhiên xung quanh trục 0 cho thấy giả định hồi quy tuyến tính phù hợp.
        """)

    # =============================================================================
    # TAB 2: MULTIPLE LINEAR REGRESSION (MLR)
    # =============================================================================
    with tab2:
        st.header("2. Hồi quy Đa biến (Multiple Linear Regression)")
        st.markdown("Dự đoán **Quality of Sleep** từ nhóm biến: **Sleep Duration**, **Stress Level**, **Heart Rate**, **Physical Activity Level**.")

        Z = df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity Level']]
        Y = df['Quality of Sleep']

        lm_mlr = LinearRegression().fit(Z, Y)
        y_pred_mlr = lm_mlr.predict(Z)

        r2_mlr = lm_mlr.score(Z, Y)
        mse_mlr = mean_squared_error(Y, y_pred_mlr)

        col_code2, col_res2 = st.columns([1, 1])

        with col_code2:
            st.subheader("💻 Code Python tính toán")
            code_tab2 = """
Z = df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity Level']]
Y = df['Quality of Sleep']

lm_mlr = LinearRegression().fit(Z, Y)
y_pred_mlr = lm_mlr.predict(Z)

print(f'MLR Intercept: {lm_mlr.intercept_:.4f}')
print(f'MLR Coefs: {lm_mlr.coef_}')
print(f'R2 score: {lm_mlr.score(Z, Y):.4f}')
print(f'MSE: {mean_squared_error(Y, y_pred_mlr):.4f}')
"""
            st.code(code_tab2, language='python')

        with col_res2:
            st.subheader("📋 Kết quả tính toán")
            st.metric("MLR Intercept", f"{lm_mlr.intercept_:.4f}")
            st.metric("R² Score", f"{r2_mlr:.4f}")
            st.metric("MSE", f"{mse_mlr:.4f}")

            coef_df = pd.DataFrame({
                'Biến độc lập': Z.columns,
                'Hệ số (Coef)': lm_mlr.coef_
            })
            st.dataframe(coef_df, hide_index=True, use_container_width=True)

        st.subheader("📊 2. Distribution Plot: Thực tế vs Dự đoán (MLR)")

        fig2 = go.Figure()
        fig2.add_trace(kde_trace(Y, 'Giá trị Thực tế (Actual)', 'red'))
        fig2.add_trace(kde_trace(y_pred_mlr, 'Dự đoán MLR (Fitted)', 'blue'))

        fig2.update_layout(
            title="2. Distribution Plot: Thực tế vs Dự đoán (MLR)",
            xaxis_title="Chất lượng Giấc ngủ (Quality of Sleep)",
            yaxis_title="Mật độ phân bố (Density)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - **Tăng độ chính xác:** Khi kết hợp cả 4 biến, **R² tăng từ {r2_slr:.4f} (SLR) lên {r2_mlr:.4f} (MLR)**.
        - **Biểu đồ phân phối:** Đường dự đoán bám theo hình dạng chung của đường thực tế nhưng chưa khớp ở mọi đỉnh; mô hình ở Tab 4 khớp sát hơn.
        """)

    # =============================================================================
    # TAB 3: POLYNOMIAL REGRESSION
    # =============================================================================
    with tab3:
        st.header("3. Hồi quy Đa thức (Polynomial Regression)")
        st.markdown("Mối quan hệ giữa **Stress Level** và **Quality of Sleep** (Đa thức Bậc 2).")

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(df[['Stress Level']])
        Y = df['Quality of Sleep']

        lm_poly = LinearRegression().fit(X_poly, Y)
        y_pred_poly = lm_poly.predict(X_poly)

        r2_poly = lm_poly.score(X_poly, Y)
        mse_poly = mean_squared_error(Y, y_pred_poly)

        col_code3, col_res3 = st.columns([1, 1])

        with col_code3:
            st.subheader("💻 Code Python tính toán")
            code_tab3 = """
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['Stress Level']])
Y = df['Quality of Sleep']

lm_poly = LinearRegression().fit(X_poly, Y)
y_pred_poly = lm_poly.predict(X_poly)

print(f'Poly Eq: Quality of Sleep = {lm_poly.intercept_:.4f} + ({lm_poly.coef_[0]:.4f}*Stress) + ({lm_poly.coef_[1]:.4f}*Stress^2)')
print(f'R2 score: {lm_poly.score(X_poly, Y):.4f}')
print(f'MSE: {mean_squared_error(Y, y_pred_poly):.4f}')
"""
            st.code(code_tab3, language='python')

        with col_res3:
            st.subheader("📋 Kết quả tính toán")
            st.metric("Phương trình Poly", f"y = {lm_poly.intercept_:.4f} + ({lm_poly.coef_[0]:.4f}*Stress) + ({lm_poly.coef_[1]:.4f}*Stress²)")
            st.metric("R² Score", f"{r2_poly:.4f}")
            st.metric("MSE", f"{mse_poly:.4f}")

        st.subheader("🌀 3. Polynomial Regression Plot")

        # Tạo dải dữ liệu mịn để vẽ đường cong
        stress_range = pd.DataFrame({'Stress Level': np.linspace(df['Stress Level'].min(), df['Stress Level'].max(), 100)})
        stress_range_poly = poly.transform(stress_range)
        y_range_pred = lm_poly.predict(stress_range_poly)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Stress Level'], y=df['Quality of Sleep'], mode='markers', name='Dữ liệu thực tế', marker=dict(color='orange', opacity=0.6)))
        fig3.add_trace(go.Scatter(x=stress_range['Stress Level'], y=y_range_pred, mode='lines', name='Đường hồi quy Đa thức Bậc 2', line=dict(color='red', width=2.5)))

        fig3.update_layout(
            title="3. Polynomial Regression: Mối quan hệ giữa Stress & Chất lượng ngủ",
            xaxis_title="Mức độ Stress (Stress Level)",
            yaxis_title="Chất lượng Giấc ngủ (Quality of Sleep)"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - Mô hình đa thức bậc 2 bắt được đường cong phi tuyến giữa mức độ Stress và Chất lượng giấc ngủ với điểm **R² = {r2_poly:.4f}**.
        """)

    # =============================================================================
    # TAB 4: PIPELINE (STANDARD SCALER + POLY DEG 2 + SLR)
    # =============================================================================
    with tab4:
        st.header("4. Pipeline (StandardScaler + Polynomial Degree 2 + LinearRegression)")
        st.markdown("Xử lý đa biến tập hợp qua Pipeline tự động hóa.")

        Z = df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity Level']]
        Y = df['Quality of Sleep']

        pipeline_input = [
            ('scale', StandardScaler()),
            ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
            ('model', LinearRegression())
        ]
        pipe = Pipeline(pipeline_input)
        pipe.fit(Z, Y)
        y_pred_pipe = pipe.predict(Z)

        r2_pipe = pipe.score(Z, Y)
        mse_pipe = mean_squared_error(Y, y_pred_pipe)

        col_code4, col_res4 = st.columns([1, 1])

        with col_code4:
            st.subheader("💻 Code Python tính toán")
            code_tab4 = """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

Z = df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity Level']]
Y = df['Quality of Sleep']

pipeline_input = [
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
]
pipe = Pipeline(pipeline_input)
pipe.fit(Z, Y)
y_pred_pipe = pipe.predict(Z)

print(f'Pipeline R2 score: {pipe.score(Z, Y):.4f}')
print(f'Pipeline MSE: {mean_squared_error(Y, y_pred_pipe):.4f}')
"""
            st.code(code_tab4, language='python')

        with col_res4:
            st.subheader("📋 Kết quả tính toán")
            st.metric("Pipeline R² Score", f"{r2_pipe:.4f}")
            st.metric("Pipeline MSE", f"{mse_pipe:.4f}")

        st.subheader("⚙️ 4. Distribution Plot: Thực tế vs Dự đoán (Pipeline Deg 2)")

        fig4 = go.Figure()
        fig4.add_trace(kde_trace(Y, 'Giá trị Thực tế (Actual)', 'red'))
        fig4.add_trace(kde_trace(y_pred_pipe, 'Dự đoán Pipeline (Fitted)', 'green'))

        fig4.update_layout(
            title="4. Distribution Plot: Thực tế vs Dự đoán (Pipeline Deg 2)",
            xaxis_title="Chất lượng Giấc ngủ (Quality of Sleep)",
            yaxis_title="Mật độ phân bố (Density)"
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - Kết hợp **Chuẩn hóa dữ liệu (StandardScaler)** và **Đa thức bậc 2 trên 4 biến** giúp mô hình đạt chỉ số **R² cao nhất ({r2_pipe:.4f})** và **MSE thấp nhất ({mse_pipe:.4f})**.
        """)