import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="StudyPulse AI - Exam Score Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Theme & CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Outfit:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, .app-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #4338ca 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 32px rgba(67, 56, 202, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: "✨";
        position: absolute;
        right: 25px;
        bottom: 10px;
        font-size: 5rem;
        opacity: 0.15;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #e0e7ff;
        margin: 0;
        max-width: 650px;
        line-height: 1.5;
    }

    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 1.6rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }

    /* Result Card Styles */
    .result-container {
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin-top: 0.5rem;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .score-badge {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -1px;
        margin: 0.8rem 0;
    }

    .tier-chip {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.35);
        margin-bottom: 0.5rem;
    }

    .grade-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 54px;
        height: 54px;
        border-radius: 50%;
        font-size: 1.5rem;
        font-weight: 800;
        background: rgba(255, 255, 255, 0.25);
        border: 2px solid white;
        margin-left: 0.8rem;
    }

    .quote-box {
        font-style: italic;
        background: rgba(0, 0, 0, 0.18);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        border-left: 3px solid rgba(255, 255, 255, 0.6);
    }

    /* Stat Pill */
    .metric-pill {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.75rem 1.1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #cbd5e1;
        font-weight: 500;
    }

    .metric-val {
        font-size: 1rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Streamlit Button Tweaks */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.65rem 1.5rem;
        transition: all 0.25s ease;
        border: none;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- MODEL LOADING ----------------- #
MODEL_PATH = "linear_regression_model.pkl"

@st.cache_resource
def load_trained_model():
    """Loads the trained Linear Regression model using joblib."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model file '{MODEL_PATH}' not found in the current directory.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_trained_model()

# Extract model parameters or use fallback
if model is not None and hasattr(model, "coef_") and hasattr(model, "intercept_"):
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
else:
    # Default parameters based on inspect if model object had issue
    slope = 9.68207815
    intercept = 2.82689235

def predict_score(hours: float) -> float:
    """Predict score using the loaded model."""
    if model is not None:
        try:
            val = float(model.predict(np.array([[hours]]))[0])
            return val
        except Exception:
            pass
    return float(slope * hours + intercept)

# ----------------- FEEDBACK & TIER LOGIC ----------------- #
def get_score_attributes(score: float, hours: float):
    """Returns aesthetic card styles, letter grade, persona title, and witty message."""
    clamped_score = min(100.0, max(0.0, score))
    
    if clamped_score >= 90:
        return {
            "tier": "Academic Weapon ⚡",
            "grade": "A+",
            "bg": "linear-gradient(135deg, #065f46 0%, #059669 50%, #10b981 100%)",
            "border": "#34d399",
            "message": "Outstanding mastery! You're operating on a different wavelength.",
            "tip": "Make sure you still get 8 hours of sleep. Your brain needs recovery to cement this knowledge!",
            "celebrate": "balloons"
        }
    elif clamped_score >= 80:
        return {
            "tier": "Honor Roll Rockstar 🌟",
            "grade": "A",
            "bg": "linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%)",
            "border": "#60a5fa",
            "message": "Phenomenal performance! You're cruising comfortably into the top percentile.",
            "tip": "Review high-weightage practice questions to turn this solid A into an effortless A+.",
            "celebrate": "balloons"
        }
    elif clamped_score >= 65:
        return {
            "tier": "Solid Contender 📚",
            "grade": "B",
            "bg": "linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #8b5cf6 100%)",
            "border": "#a78bfa",
            "message": "Good steady grasp of the material! Safe passing and respectable score.",
            "tip": "Adding just 1 to 1.5 more hours of focused revision will launch you straight into the A bracket!",
            "celebrate": None
        }
    elif clamped_score >= 50:
        return {
            "tier": "Crammer Survivor ☕",
            "grade": "C",
            "bg": "linear-gradient(135deg, #92400e 0%, #b45309 50%, #d97706 100%)",
            "border": "#f59e0b",
            "message": "You're floating right around the passing threshold. A bit risky!",
            "tip": "Try the Pomodoro technique (25 min study / 5 min break) to boost study stamina without burning out.",
            "celebrate": None
        }
    elif clamped_score >= 35:
        return {
            "tier": "Danger Zone ⚠️",
            "grade": "D",
            "bg": "linear-gradient(135deg, #991b1b 0%, #b91c1c 50%, #ef4444 100%)",
            "border": "#f87171",
            "message": "High-wire act! You're relying heavily on pure luck and multiple choice miracles.",
            "tip": "Lock away your phone, open the summary cheat-sheet, and do at least 2 more focused hours.",
            "celebrate": None
        }
    else:
        return {
            "tier": "Naptime Champion 🛋️",
            "grade": "F",
            "bg": "linear-gradient(135deg, #3f3f46 0%, #52525b 50%, #71717a 100%)",
            "border": "#a1a1aa",
            "message": "Did you study, or did you just stare at the textbook cover?",
            "tip": "Even 30-45 minutes of quick formula review will immediately rescue you from single digits.",
            "celebrate": None
        }

# ----------------- HERO BANNER ----------------- #
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
        <span style="font-size: 1.8rem;">🎯</span>
        <span style="background: rgba(255,255,255,0.2); padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Linear Regression Engine</span>
    </div>
    <h1 class="hero-title">StudyPulse AI · Score Forecaster</h1>
    <p class="hero-subtitle">
        Ever wondered what your study hours are truly worth? Input your study time, uncover your predicted test score in real time, and explore interactive grade milestones.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR ----------------- #
with st.sidebar:
    st.markdown("### 🎛️ Quick Preset Vibes")
    st.caption("Select a study scenario to test instantly:")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🛋️ Panic (1h)", use_container_width=True):
            st.session_state["study_hours_input"] = 1.0
            st.session_state["run_prediction"] = True
        if st.button("📖 Steady (6h)", use_container_width=True):
            st.session_state["study_hours_input"] = 6.0
            st.session_state["run_prediction"] = True
    with col_s2:
        if st.button("☕ Average (3.5h)", use_container_width=True):
            st.session_state["study_hours_input"] = 3.5
            st.session_state["run_prediction"] = True
        if st.button("⚡ Beast (9.5h)", use_container_width=True):
            st.session_state["study_hours_input"] = 9.5
            st.session_state["run_prediction"] = True

    st.markdown("---")
    st.markdown("### 🎯 Target Grade Reverse Calculator")
    st.caption("Tell us your goal score, and we'll calculate how long you need to study:")
    target_score = st.slider("Desired Target Score (%)", min_value=10, max_value=100, value=85, step=5)
    
    # Calculate required hours: Hours = (Score - Intercept) / Slope
    required_hours = max(0.0, (target_score - intercept) / slope)
    st.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1rem; text-align: center;">
        <span style="font-size: 0.85rem; color: #a5b4fc; font-weight: 600;">RECOMMENDED STUDY TIME</span>
        <div style="font-size: 1.9rem; font-weight: 800; color: #e0e7ff; margin: 0.2rem 0;">{required_hours:.1f} <span style="font-size: 1rem; font-weight: 500;">hours</span></div>
        <span style="font-size: 0.78rem; color: #c7d2fe;">To secure a ~{target_score}% mark</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Model DNA & Stats")
    st.markdown(f"""
    <div class="metric-pill">
        <span class="metric-label">Model Type</span>
        <span class="metric-val">Linear Regression</span>
    </div>
    <div class="metric-pill">
        <span class="metric-label">Gain per Hour studied</span>
        <span class="metric-val">+{slope:.2f} pts/hr</span>
    </div>
    <div class="metric-pill">
        <span class="metric-label">Zero-Study Base Score</span>
        <span class="metric-val">{intercept:.2f} pts</span>
    </div>
    <div class="metric-pill">
        <span class="metric-label">Equation</span>
        <span class="metric-val" style="font-size: 0.8rem;">y = {slope:.2f}x + {intercept:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

# Initialize Session State
if "study_hours_input" not in st.session_state:
    st.session_state["study_hours_input"] = 4.5
if "run_prediction" not in st.session_state:
    st.session_state["run_prediction"] = True

# ----------------- MAIN LAYOUT ----------------- #
col_input, col_display = st.columns([1.1, 1.3], gap="large")

with col_input:
    st.markdown("### ⏱️ Enter Your Study Time")
    st.write("Adjust the interactive slider or enter precise hours below:")

    # Slider control connected to session state
    hours = st.slider(
        "Hours Studied (per day/week):",
        min_value=0.0,
        max_value=12.0,
        value=float(st.session_state["study_hours_input"]),
        step=0.25,
        format="%.2f hrs",
        key="hours_slider"
    )
    
    # Sync if changed directly via slider
    st.session_state["study_hours_input"] = hours

    col_btn, col_extra = st.columns([1.3, 1])
    with col_btn:
        predict_clicked = st.button("🔮 Calculate Predicted Score", type="primary", use_container_width=True)
    with col_extra:
        auto_update = st.toggle("⚡ Real-time preview", value=True)

    # Key takeaway card
    raw_pred = predict_score(hours)
    capped_pred = min(100.0, max(0.0, raw_pred))
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 Quick Efficiency Stats")
    
    # Micro stats grid
    stat_c1, stat_c2 = st.columns(2)
    with stat_c1:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">SCORE BOOST FROM ZERO</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #38bdf8;">+{(raw_pred - intercept):.1f} pts</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_c2:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">+1 HR STUDY DIVIDEND</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #34d399;">+{slope:.1f} pts</div>
        </div>
        """, unsafe_allow_html=True)

with col_display:
    st.markdown("### 🏆 Prediction & Feedback")
    
    if predict_clicked or auto_update:
        attrs = get_score_attributes(capped_pred, hours)
        
        # Trigger celebratory animations for stellar performance
        if predict_clicked and attrs["celebrate"] == "balloons":
            st.balloons()
            
        # Display custom styled card
        st.markdown(f"""
        <div class="result-container" style="background: {attrs['bg']}; border: 2px solid {attrs['border']};">
            <div style="display: flex; justify-content: center; align-items: center;">
                <span class="tier-chip">{attrs['tier']}</span>
                <span class="grade-circle">{attrs['grade']}</span>
            </div>
            <div style="font-size: 0.9rem; letter-spacing: 1px; opacity: 0.9; text-transform: uppercase; font-weight: 600; margin-top: 0.5rem;">
                Estimated Exam Score
            </div>
            <div class="score-badge">
                {capped_pred:.1f}<span style="font-size: 2.2rem; font-weight: 600;">%</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 600; max-width: 90%; margin: 0 auto;">
                {attrs['message']}
            </div>
            <div class="quote-box">
                💡 <b>Study Coach:</b> {attrs['tip']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Custom Progress Bar with dynamic color
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.write(f"**Progress towards perfection (100%):** `{capped_pred:.1f} / 100`")
        st.progress(float(capped_pred / 100.0))

# ----------------- INTERACTIVE PLOTLY CHART ----------------- #
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("### 📈 Interactive Regression Trajectory & Milestones")
st.caption("See where your study time sits on the complete prediction curve. Hover on the line or user marker for details.")

# Generate curve data
x_curve = np.linspace(0, 12, 100)
y_curve = np.clip(slope * x_curve + intercept, 0, 100)

fig = go.Figure()

# Background grade bands
fig.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Danger Zone (<40%)", annotation_position="top left")
fig.add_hrect(y0=40, y1=65, fillcolor="orange", opacity=0.08, line_width=0, annotation_text="Passing (40-65%)", annotation_position="top left")
fig.add_hrect(y0=65, y1=85, fillcolor="blue", opacity=0.08, line_width=0, annotation_text="Merit / Honors (65-85%)", annotation_position="top left")
fig.add_hrect(y0=85, y1=100, fillcolor="green", opacity=0.08, line_width=0, annotation_text="Excellence (85-100%)", annotation_position="top left")

# Regression Line
fig.add_trace(go.Scatter(
    x=x_curve,
    y=y_curve,
    mode='lines',
    name='Regression Model',
    line=dict(color='#818cf8', width=4, shape='spline'),
    hovertemplate="<b>Study Time:</b> %{x:.2f} hrs<br><b>Predicted Score:</b> %{y:.1f}%<extra></extra>"
))

# User Selected Point
fig.add_trace(go.Scatter(
    x=[hours],
    y=[capped_pred],
    mode='markers+text',
    name='Your Position',
    text=[f"📍 You: {capped_pred:.1f}% ({hours:.1f} hrs)"],
    textposition="top center",
    textfont=dict(size=14, color="#ffffff", family="Outfit"),
    marker=dict(
        color='#fbbf24',
        size=18,
        symbol='star',
        line=dict(color='#ffffff', width=2)
    ),
    hovertemplate="<b>🌟 Your Study Time:</b> %{x:.2f} hrs<br><b>Your Predicted Score:</b> %{y:.1f}%<extra></extra>"
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    height=480,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(
        title="Hours Studied",
        dtick=1,
        range=[-0.2, 12.2],
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.2)"
    ),
    yaxis=dict(
        title="Predicted Score (%)",
        range=[0, 108],
        dtick=20,
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.2)"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)"
    ),
    hoverlabel=dict(
        bgcolor="#1e1b4b",
        font_size=13,
        font_family="Plus Jakarta Sans"
    )
)

st.plotly_chart(fig, use_container_width=True)

# ----------------- FOOTER / STUDY TIPS ----------------- #
st.markdown("<br>", unsafe_allow_html=True)
exp = st.expander("📚 Pro Study Hacks to Maximize Your Return On Time")
with exp:
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("#### 🍅 Pomodoro Cycles")
        st.write("Study with 100% focus for 25 minutes, followed by a 5-minute break. After 4 rounds, take a 20-minute rest.")
    with col_t2:
        st.markdown("#### 🧠 Active Recall")
        st.write("Don't just re-read notes passively. Test yourself with flashcards or write explanations without looking at the material.")
    with col_t3:
        st.markdown("#### 😴 Sleep Consolidation")
        st.write("Memory consolidation happens during sleep. 6 hours of study with 8 hours of sleep outperforms pulling an all-nighter every time!")

st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2.5rem; padding-bottom: 1rem;">
    Built with Streamlit & Scikit-Learn · Model equation: <code>Score = 9.68 × Hours + 2.83</code>
</div>
""", unsafe_allow_html=True)
