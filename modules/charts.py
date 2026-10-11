import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def kde_trace(values, name, color):
    # Đường mật độ (KDE) cho Distribution Plot, cùng cách tính với sns.kdeplot trong notebook
    values = np.asarray(values, dtype=float)

    # Dữ liệu chỉ có một giá trị (khi lọc quá hẹp) thì không ước lượng được mật độ
    if len(values) < 2 or np.ptp(values) == 0:
        return go.Scatter(
            x=[values[0], values[0]], y=[0, 1], mode="lines", name=name,
            line=dict(color=color, width=2),
        )

    kde = gaussian_kde(values)
    bandwidth = kde.factor * values.std(ddof=1)
    x = np.linspace(values.min() - 3 * bandwidth, values.max() + 3 * bandwidth, 200)
    return go.Scatter(
        x=x, y=kde(x), mode="lines", name=name, line=dict(color=color, width=2)
    )
