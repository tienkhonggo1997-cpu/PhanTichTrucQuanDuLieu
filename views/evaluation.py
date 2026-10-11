import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from modules.charts import kde_trace

FEATURES = ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity Level']
TARGET = 'Quality of Sleep'


def render_evaluation(df):
    st.title("📊 Chương 7: Đánh Giá & Lựa Chọn Mô Hình (Model Evaluation & Selection)")
    st.markdown(
        "Phân tích hiệu năng mô hình, hiện tượng Overfitting/Underfitting và kỹ thuật Điều chuẩn (Regularization) Ridge.")

    # Kiểm tra biến mục tiêu và các biến độc lập
    Y = df[TARGET]
    X_data = df[FEATURES]

    # Mỗi "hồ sơ" = một tổ hợp giá trị giống hệt nhau (bỏ qua Person ID).
    # Các dòng trùng hồ sơ phải nằm cùng một phía Train hoặc Test, nếu không
    # tập Test chỉ là bản sao của tập Train và R² bị thổi phồng.
    profile_cols = [c for c in df.columns if c != 'Person ID']
    groups = df[profile_cols].astype(str).agg('|'.join, axis=1)
    n_profiles = groups.nunique()

    if n_profiles < 10:
        st.warning(
            f"⚠️ Bộ lọc hiện tại chỉ còn {n_profiles} hồ sơ khác nhau, không đủ để chia Train/Test và "
            "Cross-Validation. Hãy nới rộng bộ lọc ở thanh bên.")
        return

    # Tách Train / Test (70% - 30%) theo hồ sơ
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(X_data, Y, groups))
    x_train, x_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    g_train = groups.iloc[train_idx]

    # Để minh họa: nếu chia ngẫu nhiên theo dòng thì bao nhiêu dòng Test có bản sao trong Train?
    naive_train, naive_test = train_test_split(np.arange(len(df)), test_size=0.3, random_state=42)
    n_leaked = groups.iloc[naive_test].isin(set(groups.iloc[naive_train])).sum()

    st.info(
        f"ℹ️ Dữ liệu có **{len(df)} dòng** nhưng chỉ **{n_profiles} hồ sơ khác nhau**. Nếu chia ngẫu nhiên theo dòng, "
        f"**{n_leaked}/{len(naive_test)}** dòng Test sẽ có bản sao y hệt trong Train (rò rỉ dữ liệu). "
        f"Vì vậy chương này chia theo hồ sơ: Train {len(x_train)} dòng, Test {len(x_test)} dòng, không hồ sơ nào nằm ở cả hai tập.")

    # Khởi tạo 3 Tabs dựa trên 3 phần phân tích của Chương 7
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Tab 1: Train/Test Split & Cross Validation",
        "⚖️ Tab 2: Phân tích Underfitting & Overfitting",
        "🎯 Tab 3: Hồi quy Ridge & Tối ưu Alpha",
        "🏆 Tab 4: So sánh các mô hình"
    ])

    # =============================================================================
    # TAB 1: TRAIN/TEST SPLIT & CROSS VALIDATION
    # =============================================================================
    with tab1:
        st.header("1. Đánh giá Mô hình trên Tập Test & Cross-Validation")
        st.markdown(
            "Tách tập dữ liệu theo tỉ lệ **70% Train / 30% Test** (theo hồ sơ) và đánh giá độ ổn định qua **5-Fold Cross Validation**.")

        col_code1, col_res1 = st.columns([1, 1])

        # Thực thi tính toán
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        train_r2 = lr.score(x_train, y_train)
        test_r2 = lr.score(x_test, y_test)

        cv_scores = cross_val_score(lr, X_data, Y, cv=GroupKFold(n_splits=5), groups=groups)
        cv_mean_r2 = cv_scores.mean()
        cv_std_r2 = cv_scores.std()

        with col_code1:
            st.subheader("💻 Code Python tính toán")
            code_tab1 = """
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

Y = df['Quality of Sleep']
X_data = df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity Level']]

# Các dòng giống hệt nhau (trừ Person ID) thuộc cùng một nhóm
profile_cols = [c for c in df.columns if c != 'Person ID']
groups = df[profile_cols].astype(str).agg('|'.join, axis=1)

# 1. Tách Train / Test (70% - 30%) theo nhóm
splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(splitter.split(X_data, Y, groups))
x_train, x_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

lr = LinearRegression()
lr.fit(x_train, y_train)

train_r2 = lr.score(x_train, y_train)
test_r2 = lr.score(x_test, y_test)

# 2. Cross Validation (5-Fold) theo nhóm
cv_scores = cross_val_score(lr, X_data, Y, cv=GroupKFold(n_splits=5), groups=groups)
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
        fig1.add_trace(kde_trace(y_test, 'Giá trị Thực tế (Actual)', 'red'))

        # Đường dự đoán (Predicted)
        fig1.add_trace(kde_trace(y_test_pred, 'Dự đoán Tập Test (Predicted)', 'blue'))

        fig1.update_layout(
            title="1. Đánh giá Mô hình trên Tập Test (30%)",
            xaxis_title="Quality of Sleep",
            yaxis_title="Density"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Nhận xét sinh ra từ số liệu, không viết cứng
        gap = train_r2 - test_r2
        if abs(gap) < 0.05:
            gap_text = "chênh lệch nhỏ, chưa thấy dấu hiệu overfitting ở mô hình tuyến tính 4 biến"
        elif gap > 0:
            gap_text = "Train cao hơn Test rõ rệt, là dấu hiệu mô hình đang overfitting"
        else:
            gap_text = ("Test cao hơn Train; điều này đến từ sự ngẫu nhiên của một lần chia với tập Test nhỏ, "
                        "không có nghĩa mô hình dự đoán dữ liệu mới tốt hơn dữ liệu đã học")

        if cv_std_r2 < 0.05:
            cv_text = "các fold cho kết quả gần nhau, mô hình hoạt động ổn định trên các phần dữ liệu khác nhau"
        else:
            cv_text = "kết quả dao động đáng kể giữa các fold, nên một lần chia Train/Test đơn lẻ chưa đủ tin cậy"

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - **Khả năng tổng quát hóa (Generalization):** **Train $R^2$ = {train_r2:.4f}**, **Test $R^2$ = {test_r2:.4f}** (chênh {gap:+.4f}): {gap_text}.
        - **Độ ổn định (5-Fold CV):** $R^2$ trung bình **{cv_mean_r2:.4f}** (độ lệch chuẩn {cv_std_r2:.4f}, thấp nhất {cv_scores.min():.4f}, cao nhất {cv_scores.max():.4f}): {cv_text}.
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

        # Chuẩn hóa Stress Level trước khi nâng bậc để phép tính bậc cao ổn định về mặt số học
        stress_scaler = StandardScaler().fit(x_train[['Stress Level']])
        s_train = stress_scaler.transform(x_train[['Stress Level']])
        s_test = stress_scaler.transform(x_test[['Stress Level']])

        for d in degrees:
            poly = PolynomialFeatures(degree=d)
            x_tr_p = poly.fit_transform(s_train)
            x_te_p = poly.transform(s_test)

            m = LinearRegression().fit(x_tr_p, y_train)
            r2_train.append(m.score(x_tr_p, y_train))
            r2_test.append(m.score(x_te_p, y_test))

        with col_code2:
            st.subheader("💻 Code Python tính toán")
            code_tab2 = """
degrees = [1, 2, 3, 4, 5]
r2_train, r2_test = [], []

# Chuẩn hóa Stress Level trước khi nâng bậc
stress_scaler = StandardScaler().fit(x_train[['Stress Level']])
s_train = stress_scaler.transform(x_train[['Stress Level']])
s_test = stress_scaler.transform(x_test[['Stress Level']])

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    x_tr_p = poly.fit_transform(s_train)
    x_te_p = poly.transform(s_test)

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

        best_d_idx = int(np.argmax(r2_test))
        n_stress = df['Stress Level'].nunique()
        if r2_test[-1] < r2_test[best_d_idx] - 0.02:
            over_text = (f"Sau bậc {degrees[best_d_idx]}, Train $R^2$ vẫn giữ ở mức cao nhưng Test $R^2$ giảm còn "
                         f"{r2_test[-1]:.4f} ở bậc {degrees[-1]}: đây là dấu hiệu overfitting.")
        else:
            over_text = (f"Đến bậc {degrees[-1]}, Test $R^2$ ({r2_test[-1]:.4f}) không giảm đáng kể so với mức cao nhất, "
                         "nên **chưa quan sát thấy overfitting** trong khoảng bậc này.")

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - **Bậc tốt nhất trên tập Test:** bậc **{degrees[best_d_idx]}** với Test $R^2$ = **{r2_test[best_d_idx]:.4f}** (bậc 1: {r2_test[0]:.4f}).
        - **Overfitting:** {over_text}
        - **Giới hạn của dữ liệu:** *Stress Level* chỉ nhận **{n_stress} giá trị khác nhau**, nên đa thức bậc {n_stress - 1} đã đi qua trung bình của từng mức stress; tăng bậc cao hơn không làm mô hình phức tạp thêm. Biến này vì vậy không phù hợp để minh họa overfitting mạnh.
        """)

    # =============================================================================
    # TAB 3: HỒI QUY RIDGE & TỐI ƯU ALPHA
    # =============================================================================
    with tab3:
        st.header("3. Hồi quy Ridge: Tối ưu hóa Tham số Điều chuẩn (Alpha)")
        st.markdown(
            "Ứng dụng thuật toán **Ridge Regression** kết hợp chuẩn hóa dữ liệu (`StandardScaler`) trên đa thức bậc 3. "
            "Alpha được chọn bằng **Cross-Validation trên tập Train**; tập Test chỉ dùng một lần để báo cáo kết quả cuối.")

        col_code3, col_res3 = st.columns([1, 1])

        # 1. Pipeline: Chuẩn hóa dữ liệu -> Đa thức bậc 3 -> Ridge
        ridge_pipe = Pipeline([
            ('scale', StandardScaler()),
            ('polynomial', PolynomialFeatures(degree=3, include_bias=False)),
            ('model', Ridge())
        ])

        # 2. Quét dãy alpha (Log scale) bằng 5-Fold CV theo hồ sơ, chỉ trên tập Train
        alphas = np.logspace(-2, 4, 50)
        cv_train_scores, cv_val_scores = validation_curve(
            ridge_pipe, x_train, y_train,
            param_name='model__alpha', param_range=alphas,
            cv=GroupKFold(n_splits=5), groups=g_train
        )
        r_train_scores = cv_train_scores.mean(axis=1)
        r_val_scores = cv_val_scores.mean(axis=1)

        # Tìm Alpha tối ưu theo điểm Validation
        best_idx = int(np.argmax(r_val_scores))
        best_alpha = alphas[best_idx]

        # 3. Huấn luyện lại trên toàn bộ Train với Alpha tối ưu, chấm một lần trên Test
        ridge_pipe.set_params(model__alpha=best_alpha).fit(x_train, y_train)
        ridge_train_r2 = ridge_pipe.score(x_train, y_train)
        ridge_test_r2 = ridge_pipe.score(x_test, y_test)

        with col_code3:
            st.subheader("💻 Code Python tính toán")
            code_tab3 = """
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ridge_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=3, include_bias=False)),
    ('model', Ridge())
])

# Quét Alpha bằng 5-Fold CV, chỉ trên tập Train
alphas = np.logspace(-2, 4, 50)
cv_train_scores, cv_val_scores = validation_curve(
    ridge_pipe, x_train, y_train,
    param_name='model__alpha', param_range=alphas,
    cv=GroupKFold(n_splits=5), groups=g_train
)
best_alpha = alphas[np.argmax(cv_val_scores.mean(axis=1))]

# Huấn luyện lại với Alpha tối ưu, chấm một lần trên Test
ridge_pipe.set_params(model__alpha=best_alpha).fit(x_train, y_train)
ridge_test_r2 = ridge_pipe.score(x_test, y_test)
"""
            st.code(code_tab3, language='python')

        with col_res3:
            st.subheader("📋 Kết quả Tối ưu Ridge")
            st.metric("Alpha tối ưu (Best Alpha)", f"{best_alpha:.4f}")
            st.metric("Validation R² (CV trên Train)", f"{r_val_scores[best_idx]:.4f}")
            st.metric("Test R² Đạt được", f"{ridge_test_r2:.4f}")
            st.metric("Train R² Tương ứng", f"{ridge_train_r2:.4f}")

        st.subheader("📊 Biểu đồ Mối liên quan giữa Alpha (Log Scale) và R²")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=alphas, y=r_train_scores, mode='lines', name='Train R² (trung bình CV)',
                                  line=dict(color='blue')))
        fig3.add_trace(go.Scatter(x=alphas, y=r_val_scores, mode='lines', name='Validation R² (trung bình CV)',
                                  line=dict(color='red', width=2)))

        # Vạch đường thẳng alpha tối ưu
        y_low = min(r_train_scores.min(), r_val_scores.min())
        y_high = max(r_train_scores.max(), r_val_scores.max())
        fig3.add_trace(go.Scatter(x=[best_alpha, best_alpha], y=[y_low, y_high], mode='lines',
                                  name=f'Alpha tối ưu ({best_alpha:.2f})',
                                  line=dict(color='green', dash='dash')))

        fig3.update_layout(
            title="3. Hồi quy Ridge: Mối liên quan giữa Alpha và R²",
            xaxis_title="Hệ số điều chuẩn Alpha (Log scale)",
            yaxis_title="Hệ số xác định R²",
            xaxis_type="log"
        )
        st.plotly_chart(fig3, use_container_width=True)

        diff = ridge_test_r2 - test_r2
        if diff > 0.02:
            compare_text = "ở lần chia này Ridge bậc 3 **tốt hơn** mô hình tuyến tính trên dữ liệu chưa từng thấy"
        elif diff < -0.02:
            compare_text = ("Ridge bậc 3 **kém hơn** mô hình tuyến tính trên dữ liệu chưa từng thấy: độ phức tạp tăng thêm "
                            "không cải thiện khả năng dự đoán, nên chọn mô hình tuyến tính đơn giản hơn")
        else:
            compare_text = ("hai mô hình **tương đương** trên dữ liệu chưa từng thấy, nên ưu tiên mô hình tuyến tính "
                            "vì đơn giản và dễ giải thích hơn")

        if best_idx in (0, len(alphas) - 1):
            edge_text = (" Alpha tối ưu nằm ở **biên của dải khảo sát**, tức đường Validation chưa đạt đỉnh trong dải này; "
                         "cần mở rộng dải alpha nếu muốn kết luận chắc hơn.")
        else:
            edge_text = ""

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - **Vai trò của Alpha:** Khi $\\alpha$ quá nhỏ, mô hình giống như Hồi quy đa thức thông thường (dễ bị Overfitting). Khi $\\alpha$ quá lớn, mô hình bị phạt trọng số quá mạnh dẫn đến Underfitting.
        - **Giá trị tối ưu:** $\\alpha = {best_alpha:.4f}$ cho Validation $R^2$ cao nhất (**{r_val_scores[best_idx]:.4f}**).{edge_text}
        - **So sánh trên tập Test:** Ridge bậc 3 đạt Test $R^2$ = **{ridge_test_r2:.4f}** (Train {ridge_train_r2:.4f}), mô hình tuyến tính 4 biến ở Tab 1 đạt **{test_r2:.4f}**. Kết luận: {compare_text}.
        - **Lưu ý:** đây là kết quả của **một lần chia** với tập Test {len(x_test)} dòng. Thứ hạng các mô hình có thể thay đổi khi chia khác đi; xem Tab 4 để so sánh qua kiểm định chéo và qua nhiều lần chia.
        """)

    # =============================================================================
    # TAB 4: SO SÁNH CÁC MÔ HÌNH
    # =============================================================================
    with tab4:
        st.header("4. So sánh các mô hình hồi quy")
        st.markdown(
            "Đánh giá bốn mô hình của Chương 6 và mô hình Ridge trên **cùng một cách chia theo hồ sơ**, "
            "để so sánh công bằng khả năng dự đoán dữ liệu chưa từng thấy.")

        def poly_pipe(degree, model):
            return Pipeline([
                ('scale', StandardScaler()),
                ('polynomial', PolynomialFeatures(degree=degree, include_bias=False)),
                ('model', model)
            ])

        # Tên mô hình -> (các biến đầu vào, mô hình)
        models = {
            'Tuyến tính đơn (Sleep Duration)': (['Sleep Duration'], LinearRegression()),
            'Đa thức bậc 2 (Stress Level)': (['Stress Level'], poly_pipe(2, LinearRegression())),
            'Tuyến tính đa biến (4 biến)': (FEATURES, LinearRegression()),
            'Pipeline đa thức bậc 2 (4 biến)': (FEATURES, poly_pipe(2, LinearRegression())),
            'Ridge đa thức bậc 3 (4 biến)': (FEATURES, poly_pipe(3, Ridge(alpha=best_alpha))),
        }

        # 1. Một lần chia Train/Test + 5-Fold CV theo hồ sơ
        rows = []
        for name, (cols, est) in models.items():
            m = clone(est).fit(x_train[cols], y_train)
            cv = cross_val_score(clone(est), X_data[cols], Y, cv=GroupKFold(n_splits=5), groups=groups)
            rows.append({
                'Mô hình': name,
                'Train R²': m.score(x_train[cols], y_train),
                'Test R²': m.score(x_test[cols], y_test),
                'Test MSE': mean_squared_error(y_test, m.predict(x_test[cols])),
                'CV R² (TB)': cv.mean(),
                'CV R² (thấp nhất)': cv.min(),
            })
        compare_df = pd.DataFrame(rows)

        # 2. Độ ổn định: lặp lại 30 cách chia Train/Test khác nhau
        stability = {name: [] for name in models}
        repeat_split = GroupShuffleSplit(n_splits=30, test_size=0.3, random_state=1)
        for tr_idx, te_idx in repeat_split.split(X_data, Y, groups):
            for name, (cols, est) in models.items():
                m = clone(est).fit(X_data.iloc[tr_idx][cols], Y.iloc[tr_idx])
                stability[name].append(m.score(X_data.iloc[te_idx][cols], Y.iloc[te_idx]))
        stability_df = pd.DataFrame(stability)
        stability_summary = pd.DataFrame({
            'Mô hình': stability_df.columns,
            'Test R² trung bình': stability_df.mean().values,
            'Thấp nhất': stability_df.min().values,
            'Cao nhất': stability_df.max().values,
            'Độ lệch chuẩn': stability_df.std().values,
        })

        col_code4, col_res4 = st.columns([1, 1])

        with col_code4:
            st.subheader("💻 Code Python tính toán")
            code_tab4 = """
models = {
    'Tuyến tính đơn (Sleep Duration)': (['Sleep Duration'], LinearRegression()),
    'Đa thức bậc 2 (Stress Level)': (['Stress Level'], poly_pipe(2, LinearRegression())),
    'Tuyến tính đa biến (4 biến)': (FEATURES, LinearRegression()),
    'Pipeline đa thức bậc 2 (4 biến)': (FEATURES, poly_pipe(2, LinearRegression())),
    'Ridge đa thức bậc 3 (4 biến)': (FEATURES, poly_pipe(3, Ridge(alpha=best_alpha))),
}

for name, (cols, est) in models.items():
    m = clone(est).fit(x_train[cols], y_train)
    train_r2 = m.score(x_train[cols], y_train)
    test_r2 = m.score(x_test[cols], y_test)
    cv = cross_val_score(clone(est), X_data[cols], Y,
                         cv=GroupKFold(n_splits=5), groups=groups)
"""
            st.code(code_tab4, language='python')

        with col_res4:
            st.subheader("📋 Kết quả trên một lần chia & Cross-Validation")
            st.dataframe(compare_df.round(4), hide_index=True, use_container_width=True)

        st.subheader("📊 Biểu đồ so sánh R² của các mô hình")

        fig4 = go.Figure()
        for col, color in [('Train R²', 'blue'), ('Test R²', 'red'), ('CV R² (TB)', 'green')]:
            fig4.add_trace(go.Bar(x=compare_df['Mô hình'], y=compare_df[col], name=col, marker_color=color,
                                  opacity=0.75, text=compare_df[col].round(3), textposition='outside'))
        fig4.update_layout(
            title="4. So sánh R² huấn luyện, kiểm tra và kiểm định chéo",
            yaxis_title="Hệ số xác định R²",
            barmode='group'
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("🔁 Độ ổn định qua 30 lần chia Train/Test")
        st.dataframe(stability_summary.round(4), hide_index=True, use_container_width=True)

        best_cv = compare_df.loc[compare_df['CV R² (TB)'].idxmax()]
        best_test = compare_df.loc[compare_df['Test R²'].idxmax()]
        best_stable = stability_summary.loc[stability_summary['Test R² trung bình'].idxmax()]
        single_best = compare_df.iloc[:2]['CV R² (TB)'].max()
        multi_linear = compare_df.iloc[2]['CV R² (TB)']

        st.subheader("💡 Giải thích & Phân tích")
        st.markdown(f"""
        - **Mô hình tốt nhất theo Cross-Validation:** **{best_cv['Mô hình']}** với $R^2$ trung bình **{best_cv['CV R² (TB)']:.4f}**; trên một lần chia, Test $R^2$ cao nhất thuộc về **{best_test['Mô hình']}** ({best_test['Test R²']:.4f}).
        - **Một biến so với nhiều biến:** hai mô hình một biến đạt $R^2$ kiểm định chéo cao nhất là {single_best:.4f}, trong khi mô hình tuyến tính 4 biến đạt {multi_linear:.4f}: thêm biến đầu vào cải thiện rõ khả năng dự đoán.
        - **Độ ổn định:** qua 30 lần chia, **{best_stable['Mô hình']}** có Test $R^2$ trung bình cao nhất (**{best_stable['Test R² trung bình']:.4f}**, thấp nhất {best_stable['Thấp nhất']:.4f}, độ lệch chuẩn {best_stable['Độ lệch chuẩn']:.4f}).
        - **Lưu ý:** alpha của Ridge được chọn một lần ở Tab 3 và giữ cố định trong các phép so sánh này.
        """)
