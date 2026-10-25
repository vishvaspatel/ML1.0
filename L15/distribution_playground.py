import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

st.set_page_config(page_title="Distribution Comparison", page_icon="📊", layout="wide")
st.title("📊 Distribution Comparison")
st.caption("Compare normal distributions, convert scores to z-scores, and explore a log-normal curve.")

normal_tab, log_tab = st.tabs(["Normal & Z-scores", "Log-normal"])

with normal_tab:
    st.subheader("1. Compare two colleges")
    st.write("Change the values and compare the students relative to their own college.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### College A")
        mean_a = st.slider("Average mark", 0, 100, 70, key="mean_a")
        sd_a = st.slider("Standard deviation", 1, 25, 10, key="sd_a")
        score_a = st.slider("Student's mark", 0, 100, 85, key="score_a")
    with right:
        st.markdown("#### College B")
        mean_b = st.slider("Average mark", 0, 100, 50, key="mean_b")
        sd_b = st.slider("Standard deviation", 1, 25, 5, key="sd_b")
        score_b = st.slider("Student's mark", 0, 100, 60, key="score_b")

    z_a = (score_a - mean_a) / sd_a
    z_b = (score_b - mean_b) / sd_b

    st.markdown("### Compare the graphs")
    chart_left, chart_right = st.columns(2)

    # Original distributions
    with chart_left:
        st.markdown("**A. Original marks**")
        lower = min(mean_a - 4 * sd_a, mean_b - 4 * sd_b, 0)
        upper = max(mean_a + 4 * sd_a, mean_b + 4 * sd_b, 100)
        x = np.linspace(lower, upper, 600)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(x, norm.pdf(x, mean_a, sd_a), label="College A")
        ax.plot(x, norm.pdf(x, mean_b, sd_b), label="College B")
        ax.axvline(score_a, linestyle="--", label=f"A student's mark: {score_a}")
        ax.axvline(score_b, linestyle=":", label=f"B student's mark: {score_b}")
        ax.set_xlabel("Marks")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Standard normal scale, marking both students' z-scores
    with chart_right:
        st.markdown("**B. Standardized scale (z-scores)**")
        xz = np.linspace(-4, 4, 600)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(xz, norm.pdf(xz), label="Standard normal curve")
        ax.axvline(z_a, linestyle="--", label=f"College A: z = {z_a:.2f}")
        ax.axvline(z_b, linestyle=":", label=f"College B: z = {z_b:.2f}")
        ax.set_xlabel("Z-score")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("### Student comparison")
    st.latex(r"z = \frac{\text{student's mark} - \text{college average}}{\text{standard deviation}}")
    st.table({
        "College": ["A", "B"],
        "Student's mark": [score_a, score_b],
        "Average": [mean_a, mean_b],
        "SD": [sd_a, sd_b],
        "Z-score": [round(z_a, 2), round(z_b, 2)],
    })
    if z_a > z_b:
        st.success("College A's student is farther above their college average.")
    elif z_b > z_a:
        st.success("College B's student is farther above their college average.")
    else:
        st.info("Both students are equally far above their college averages.")

    st.caption("A larger z-score means a stronger result relative to that student's own college, assuming the distributions are comparable and approximately normal.")

with log_tab:
    st.subheader("2. Explore a log-normal distribution")
    st.write("A log-normal distribution contains only positive values and usually has a long tail to the right.")

    c1, c2 = st.columns(2)
    with c1:
        median = st.slider("Median / typical value", 1, 100, 20)
    with c2:
        sigma = st.slider("Spread (σ)", 0.1, 1.2, 0.5, 0.1)

    # For a log-normal distribution, exp(mu) is the median.
    mu = np.log(median)
    xmax = max(median * 4, np.exp(mu + 4 * sigma))
    x = np.linspace(0.001, xmax, 700)
    y = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    mean_x = np.exp(mu + sigma**2 / 2)
    mode_x = np.exp(mu - sigma**2)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, label="Log-normal curve")
    ax.axvline(mode_x, linestyle=":", label=f"Mode = {mode_x:.1f}")
    ax.axvline(median, linestyle="--", label=f"Median = {median:.1f}")
    ax.axvline(mean_x, linestyle="-.", label=f"Mean = {mean_x:.1f}")
    ax.set_xlabel("Value (X)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Mode", f"{mode_x:.2f}")
    b.metric("Median", f"{median:.2f}")
    c.metric("Mean", f"{mean_x:.2f}")
    st.caption("Try increasing σ. The curve becomes more spread out and more right-skewed. Example applications sometimes include incomes, house prices, and completion times; real data should be checked before using this model.")

st.divider()
st.caption("Teaching note: these are theoretical curves, not real student data. Standardization does not turn non-normal data into normal data.")
