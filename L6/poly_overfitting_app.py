"""Streamlit app: polynomial regression overfitting demo.

Run with:  streamlit run poly_overfitting_app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Polynomial Regression Overfitting", layout="wide")

st.title("Polynomial Regression: Watching a Model Overfit")
st.caption(
    "A few noisy points are generated from a simple curve. Increase the polynomial "
    "degree and watch the fitted curve wiggle through every training point while "
    "the true pattern gets lost."
)

with st.sidebar:
    st.header("Controls")
    n_points = st.slider("Number of data points", min_value=10, max_value=50, value=20)
    noise_sd = st.slider("Noise level", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    max_degree = st.slider("Polynomial degree", min_value=1, max_value=25, value=1)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

rng = np.random.default_rng(seed)
x_all = np.linspace(0, 10, n_points)
y_true_all = 3 + 2 * np.sin(x_all)
y_all = y_true_all + rng.normal(0, noise_sd, n_points)

# Fixed train/test split so only the degree changes between reruns.
shuffled = np.random.default_rng(seed).permutation(n_points)
n_test = max(3, n_points // 3)
test_idx, train_idx = shuffled[:n_test], shuffled[n_test:]
x_train, y_train = x_all[train_idx], y_all[train_idx]
x_test, y_test = x_all[test_idx], y_all[test_idx]

x_fit = np.linspace(0, 10, 300)

# Fix the y-range to the data's natural spread so a wild high-degree fit
# visibly shoots off the chart instead of the axes auto-zooming out to hide it.
y_margin = 3 * max(noise_sd, 1.0)
y_range = [y_all.min() - y_margin, y_all.max() + y_margin]

# Raw x in [0, 10] raised to a high power creates a badly-conditioned design
# matrix, which makes lstsq quietly damp the fit instead of overfitting.
# Rescaling to [-1, 1] keeps the polynomial features numerically well-behaved.
def scale_x(values):
    return (values - 5.0) / 5.0


def fit_degree(degree):
    poly = PolynomialFeatures(degree=degree)
    X_train = poly.fit_transform(scale_x(x_train).reshape(-1, 1))
    X_test = poly.transform(scale_x(x_test).reshape(-1, 1))
    X_fit = poly.transform(scale_x(x_fit).reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return model.predict(X_fit), train_r2, test_r2


y_pred_fit, train_r2, test_r2 = fit_degree(max_degree)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_train, y=y_train, mode="markers", name="Training data", marker=dict(size=10, color="black")))
fig.add_trace(go.Scatter(x=x_test, y=y_test, mode="markers", name="Held-out (test) data", marker=dict(size=10, color="orange", symbol="diamond")))
fig.add_trace(go.Scatter(x=x_fit, y=y_pred_fit, mode="lines", name=f"Degree {max_degree} fit", line=dict(color="red", width=3)))
fig.add_trace(go.Scatter(x=x_fit, y=3 + 2 * np.sin(x_fit), mode="lines", name="True pattern", line=dict(color="green", dash="dash")))
fig.update_layout(xaxis_title="x", yaxis_title="y", height=550, yaxis_range=y_range)

st.plotly_chart(fig, width="stretch")

col1, col2, col3 = st.columns(3)
col1.metric("Training R²", f"{train_r2:.3f}")
col2.metric("Test R² (held-out)", f"{test_r2:.3f}")
col3.metric("Polynomial degree", max_degree)

# Sweep every degree up to the chosen one to show the classic overfitting curve:
# train R² keeps climbing while test R² eventually falls off a cliff.
degrees = list(range(1, max_degree + 1))
train_scores, test_scores = [], []
for d in degrees:
    _, tr, te = fit_degree(d)
    train_scores.append(tr)
    test_scores.append(te)

trend_fig = go.Figure()
trend_fig.add_trace(go.Scatter(x=degrees, y=train_scores, mode="lines+markers", name="Train R²", line=dict(color="black")))
trend_fig.add_trace(go.Scatter(x=degrees, y=test_scores, mode="lines+markers", name="Test R²", line=dict(color="orange")))
trend_fig.update_layout(xaxis_title="Polynomial degree", yaxis_title="R²", height=350, yaxis_range=[min(-1, min(test_scores)), 1.05])
st.subheader("Train vs. test R² as degree increases")
st.plotly_chart(trend_fig, width="stretch")

if test_r2 < 0.3 and train_r2 > 0.9:
    st.warning(
        "Training R² is near-perfect while test R² has collapsed - the model has "
        "memorized the training points instead of learning the true pattern."
    )
elif max_degree >= len(x_train) - 1:
    st.warning(
        "Degree is close to (or exceeds) the number of training points: the curve can "
        "pass through almost every training point, which is a classic overfitting sign."
    )
