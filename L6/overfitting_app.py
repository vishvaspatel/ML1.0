"""Streamlit app: seeing overfitting as a LINE in 2D and a PLANE in 3D.

Run with:  streamlit run overfitting_app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Overfitting in 2D and 3D", layout="wide")


def adjusted_r2(r2, n, k):
    """Adjusted R2 = 1 - [ (1 - R2) * (n - 1) / (n - k - 1) ]."""
    if n <= k + 1:
        raise ValueError(f"Adjusted R2 needs n > k + 1, but got n={n} and k={k}.")
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


@st.cache_data
def make_data(seed, n_train, n_test, noise_sd):
    """study_hours drives marks; shoe_size is pure noise and is never used to build marks."""

    def draw(generator, count):
        study_hours = generator.uniform(1.0, 10.0, count)
        marks = 30 + 6 * study_hours + generator.normal(0.0, noise_sd, count)
        shoe_size = generator.normal(8.0, 1.5, count)
        return study_hours, shoe_size, marks

    # Separate streams so changing one group never reshuffles the other.
    train = draw(np.random.default_rng(seed), n_train)
    test = draw(np.random.default_rng(seed + 10_000), n_test)
    return train + test


st.title("Why training R² goes up when you add a useless feature")
st.caption(
    "Model A sees only `study_hours` and can draw a **line**. "
    "Model B also sees `shoe_size` (pure random noise) and can draw a **plane**. "
    "A plane can always choose to stay flat, so it can never fit the training data worse - "
    "but on **unseen students** there is no such protection."
)

with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=2025, step=1)
    n_points = st.slider("Number of training students", min_value=5, max_value=60, value=14)
    n_test = st.slider("Number of unseen test students", min_value=10, max_value=200, value=60)
    st.caption("Change the seed to draw a fresh class of students - the effect shows up again and again.")
    noise_sd = st.slider("Noise in marks (std dev)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
    st.divider()
    st.subheader("3D view")
    show_flat_reference = st.checkbox("Show the old flat sheet in grey on the right", value=True)
    show_sticks = st.checkbox("Show the red error sticks", value=True)
    show_test_points = st.checkbox("Show the unseen test students", value=False)
    st.caption("Drag inside a 3D chart to rotate it, scroll to zoom, double-click to reset.")

x1, x2, y_true, x1_te, x2_te, y_te = make_data(int(seed), int(n_points), int(n_test), float(noise_sd))
X_a = x1.reshape(-1, 1)
X_b = np.column_stack([x1, x2])
X_a_te = x1_te.reshape(-1, 1)
X_b_te = np.column_stack([x1_te, x2_te])

# Model A - one feature. The fitted shape is a LINE living in a 2D world.
model_a = LinearRegression().fit(X_a, y_true)
pred_a = model_a.predict(X_a)

# Model B - two features. The fitted shape is a PLANE living in a 3D world.
model_b = LinearRegression().fit(X_b, y_true)
pred_b = model_b.predict(X_b)

resid_a = y_true - pred_a
resid_b = y_true - pred_b
sse_a = float(np.sum(resid_a**2))
sse_b = float(np.sum(resid_b**2))
r2_a = r2_score(y_true, pred_a)
r2_b = r2_score(y_true, pred_b)
adj_a = adjusted_r2(r2_a, n_points, 1)
adj_b = adjusted_r2(r2_b, n_points, 2)

# The honest exam: students the models have never seen.
test_r2_a = r2_score(y_te, model_a.predict(X_a_te))
test_r2_b = r2_score(y_te, model_b.predict(X_b_te))
test_rmse_a = float(np.sqrt(np.mean((y_te - model_a.predict(X_a_te)) ** 2)))
test_rmse_b = float(np.sqrt(np.mean((y_te - model_b.predict(X_b_te)) ** 2)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training R²", f"{r2_b:.4f}", f"{r2_b - r2_a:+.4f}")
c2.metric("TEST R²", f"{test_r2_b:.4f}", f"{test_r2_b - test_r2_a:+.4f}")
c3.metric("Training Adjusted R²", f"{adj_b:.4f}", f"{adj_b - adj_a:+.4f}")
c4.metric("Coefficient of shoe_size", f"{model_b.coef_[1]:.4f}", "true value is 0", delta_color="off")
st.caption("Each number is **Model B**, and the arrow underneath is the change caused by adding `shoe_size`.")

score_table = pd.DataFrame(
    {
        "Model A (study_hours only)": [r2_a, adj_a, sse_a, test_r2_a, test_rmse_a],
        "Model B (+ shoe_size)": [r2_b, adj_b, sse_b, test_r2_b, test_rmse_b],
        "Change": [
            r2_b - r2_a,
            adj_b - adj_a,
            sse_b - sse_a,
            test_r2_b - test_r2_a,
            test_rmse_b - test_rmse_a,
        ],
    },
    index=["Training R²", "Training Adjusted R²", "Training SSE", "TEST R²", "TEST RMSE"],
)
st.table(score_table.round(4))

if r2_b >= r2_a and test_r2_b < test_r2_a:
    st.error(
        f"Training R² went **UP** by {r2_b - r2_a:+.4f} while TEST R² went **DOWN** by {test_r2_b - test_r2_a:+.4f}. "
        "The model looks better on the data it memorised and is actually worse on new students - that is overfitting."
    )
elif test_r2_b >= test_r2_a:
    st.warning(
        f"On this particular seed the junk feature happened to help on the test set too ({test_r2_b - test_r2_a:+.4f}). "
        "That is luck, not skill - try a few other seeds and you will see the test score fall far more often than it rises."
    )

st.divider()

# --------------------------------------------------------------------------
# 2D: one feature -> a line
# --------------------------------------------------------------------------
st.subheader("1 feature → the model can only draw a LINE (2D)")

grid = np.linspace(x1.min() - 0.4, x1.max() + 0.4, 100)
line_a = model_a.intercept_ + model_a.coef_[0] * grid

fig2d, ax = plt.subplots(figsize=(10, 6))
for xi, yi, pi in zip(x1, y_true, pred_a):
    ax.plot([xi, xi], [yi, pi], color="red", linewidth=1.6, zorder=1)
ax.plot(
    grid,
    line_a,
    color="crimson",
    linewidth=2.5,
    zorder=2,
    label=f"Model A line: marks = {model_a.intercept_:.2f} + {model_a.coef_[0]:.2f} * study_hours",
)
ax.scatter(x1, y_true, s=110, color="#1f77b4", edgecolor="white", zorder=3, label="Real training students")
ax.scatter(x1, pred_a, s=70, marker="x", color="black", linewidth=1.8, zorder=4, label="Model A prediction")
ax.set_xlabel("study_hours  (the only feature Model A can see)")
ax.set_ylabel("marks")
ax.set_title(
    "With ONE feature the model may only draw a LINE\n"
    f"training R² = {r2_a:.4f}   |   SSE = {sse_a:.2f}   |   red sticks = the leftover error"
)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig2d.tight_layout()
st.pyplot(fig2d)
plt.close(fig2d)

st.info(
    "The red sticks are the training errors. This is already the **best possible line** - "
    "no other line makes them shorter. To do better the model needs a brand new direction to move in."
)

st.divider()

# --------------------------------------------------------------------------
# 3D: two features -> a plane that is allowed to tilt
# --------------------------------------------------------------------------
st.subheader("2 features → the model draws a PLANE that is allowed to TILT (3D)")
st.caption(
    "**Drag** inside either chart to rotate it, **scroll** to zoom, **double-click** to reset the view. "
    "Spin the right-hand plane around and watch how it leans along the `shoe_size` axis, "
    "while the left one stays perfectly level."
)

# Both models are flat surfaces, so a 2x2 grid is enough to draw them as one clean sheet.
g1 = np.linspace(x1.min() - 0.4, x1.max() + 0.4, 2)
g2 = np.linspace(x2.min() - 0.4, x2.max() + 0.4, 2)
G1, G2 = np.meshgrid(g1, g2)

ZA = model_a.intercept_ + model_a.coef_[0] * G1
ZB = model_b.intercept_ + model_b.coef_[0] * G1 + model_b.coef_[1] * G2

# Identical z-limits in both panels, otherwise the red sticks cannot be compared by eye.
z_low = float(min(y_true.min(), ZA.min(), ZB.min()) - 3)
z_high = float(max(y_true.max(), ZA.max(), ZB.max()) + 3)


def flat_colorscale(color):
    return [[0.0, color], [1.0, color]]


# Kill Plotly's 3D shading so each sheet keeps its true, flat colour.
FLAT_LIGHTING = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)


def error_sticks(pred):
    """One trace holding every vertical error segment, separated by None gaps."""
    sx, sy, sz = [], [], []
    for xi, si, yi, pi in zip(x1, x2, y_true, pred):
        sx += [xi, xi, None]
        sy += [si, si, None]
        sz += [yi, pi, None]
    return go.Scatter3d(
        x=sx, y=sy, z=sz, mode="lines",
        line=dict(color="red", width=5),
        name="training error", hoverinfo="skip",
    )


def build_scene(Z, pred, sse, r2, test_r2, color, surface_name, add_flat_reference):
    fig = go.Figure()

    if add_flat_reference:
        fig.add_trace(go.Surface(
            x=g1, y=g2, z=ZA, colorscale=flat_colorscale("#9e9e9e"), showscale=False,
            opacity=0.20, name="old flat sheet", showlegend=True, hoverinfo="skip",
            lighting=FLAT_LIGHTING,
        ))

    fig.add_trace(go.Surface(
        x=g1, y=g2, z=Z, colorscale=flat_colorscale(color), showscale=False,
        opacity=0.80, name=surface_name, showlegend=True, lighting=FLAT_LIGHTING,
        hovertemplate="study_hours %{x:.2f}<br>shoe_size %{y:.2f}<br>predicted marks %{z:.1f}<extra></extra>",
    ))

    if show_sticks:
        fig.add_trace(error_sticks(pred))

    fig.add_trace(go.Scatter3d(
        x=x1, y=x2, z=y_true, mode="markers",
        marker=dict(size=5, color="#08306b", line=dict(color="white", width=1)),
        name="training students",
        hovertemplate="study_hours %{x:.2f}<br>shoe_size %{y:.2f}<br>actual marks %{z:.1f}<extra></extra>",
    ))

    if show_test_points:
        fig.add_trace(go.Scatter3d(
            x=x1_te, y=x2_te, z=y_te, mode="markers",
            marker=dict(size=4, color="#2ca02c", symbol="diamond", opacity=0.85),
            name="unseen test students",
            hovertemplate="study_hours %{x:.2f}<br>shoe_size %{y:.2f}<br>actual marks %{z:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"{surface_name}<br><sub>training R² = {r2:.4f}   |   test R² = {test_r2:.4f}   |   SSE = {sse:.2f}</sub>",
            x=0.02,
        ),
        scene=dict(
            xaxis_title="study_hours (useful)",
            yaxis_title="shoe_size (irrelevant)",
            zaxis_title="marks",
            zaxis=dict(range=[z_low, z_high]),
            aspectratio=dict(x=1, y=1, z=0.75),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, x=0),
    )
    return fig


left, right = st.columns(2)
with left:
    st.plotly_chart(
        build_scene(ZA, pred_a, sse_a, r2_a, test_r2_a, "#6baed6",
                    "Model A: FLAT sheet - shoe_size ignored", False),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        build_scene(ZB, pred_b, sse_b, r2_b, test_r2_b, "#ff7f0e",
                    "Model B: TILTED plane - shoe_size used", show_flat_reference),
        use_container_width=True,
    )

improved = int(np.sum(np.abs(resid_b) < np.abs(resid_a)))
st.success(
    f"**{improved} of the {n_points} training points** moved closer to the prediction surface. "
    f"Training SSE dropped by **{sse_a - sse_b:.2f}** ({100 * (sse_a - sse_b) / sse_a:.2f}%), so training R² rose by "
    f"**{r2_b - r2_a:+.4f}** - even though `shoe_size` carries no information at all."
)

st.divider()

# --------------------------------------------------------------------------
# Training vs test: the score that is allowed to fall
# --------------------------------------------------------------------------
st.subheader("Training R² can only rise - TEST R² is free to fall")

fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4.5))
positions = np.arange(2)
bar_width = 0.36
train_bars = ax_cmp.bar(positions - bar_width / 2, [r2_a, r2_b], bar_width,
                        color="#2ca02c", edgecolor="black", linewidth=0.5, label="Training R²")
test_bars = ax_cmp.bar(positions + bar_width / 2, [test_r2_a, test_r2_b], bar_width,
                       color="#1f77b4", edgecolor="black", linewidth=0.5,
                       label=f"Test R² ({n_test} unseen students)")
for group in (train_bars, test_bars):
    ax_cmp.bar_label(group, fmt="%.4f", padding=2, fontsize=9)

ax_cmp.set_xticks(positions)
ax_cmp.set_xticklabels(["Model A\n(study_hours)", "Model B\n(+ shoe_size)"])
ax_cmp.set_ylabel("R²")
ax_cmp.set_ylim(0, max(r2_a, r2_b, test_r2_a, test_r2_b) * 1.25)
ax_cmp.set_title("Green always grows or stays level. Blue is the one that tells the truth.")
ax_cmp.grid(axis="y", alpha=0.3)
ax_cmp.legend(loc="lower left", fontsize=9)
fig_cmp.tight_layout()
st.pyplot(fig_cmp)
plt.close(fig_cmp)

st.markdown(
    f"""
| | Training R² | Test R² |
|---|---|---|
| Model A - `study_hours` | {r2_a:.4f} | {test_r2_a:.4f} |
| Model B - `+ shoe_size` | {r2_b:.4f} | {test_r2_b:.4f} |
| **Change** | **{r2_b - r2_a:+.4f}** | **{test_r2_b - test_r2_a:+.4f}** |

The training bar is scored on the very rows the coefficients were tuned on, so the extra feature is **guaranteed**
not to hurt there. The test bar is scored on students the model has never met, where the accidental tilt it learned
is just noise - so that bar is free to drop, and usually does.
"""
)

st.markdown(
    """
**Why this happens**

- The plane could always have stayed **flat** (coefficient of `shoe_size` = `0`), which reproduces the line exactly.
- Least squares picks the *best* option, so the plane can never fit the training data worse → **training R² cannot decrease**.
- Because `shoe_size` is random, some slight tilt always happens to hug *these particular* students a bit better, so the plane tilts and training R² actually goes **up**.
- That tilt was fitted to this sample's noise. A new student with a different shoe size gets pushed up or down by a rule that does not exist → **test R² gets worse**.
- Adjusted R² charges a fee for the extra feature, so it usually **goes down** - it is a cheap warning sign you can compute without a test set.
- **The lesson:** never judge a feature by the training score. Tick *"Show the unseen test students"* in the sidebar to watch the tilted plane miss the green diamonds it was never fitted on.
"""
)
