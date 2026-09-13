import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")

def plot_line_explorer(m, b):
    """Plots the y = mx + b line with fixed axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-10, 10, 100)
    y = m * x + b
    
    ax.plot(x, y, color='blue', linewidth=2)
    
    # Fixed axes to make the effect of m and b visually intuitive
    ax.set_xlim([-10, 10])
    ax.set_ylim([-50, 50])
    
    # Grid and origin lines
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("y = mx + b")
    return fig

def train_and_evaluate(df):
    """Trains a linear regression model and returns model, metrics, and predictions."""
    X = df[['CGPA']].values
    y = df['Placement_LPA'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    return model, r2, mse

def plot_regression(df, model, history=None, selected_runs=None):
    """Plots the scatter data and the fitted regression line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    X = df[['CGPA']].values
    y = df['Placement_LPA'].values
    
    # Generate points for the regression line
    x_range = np.linspace(df['CGPA'].min() - 0.5, df['CGPA'].max() + 0.5, 100)
    
    # Plot historical lines first
    if history and selected_runs:
        for run_id in selected_runs:
            run_data = next((item for item in history if item["Run #"] == run_id), None)
            if run_data:
                m_hist = run_data["m"]
                b_hist = run_data["b"]
                y_hist = m_hist * x_range + b_hist
                ax.plot(x_range, y_hist, linestyle='--', alpha=0.6, label=f'Run {run_id} (y={m_hist:.1f}x+{b_hist:.1f})')
                
    # Plot current regression line
    y_range_pred = model.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_range_pred, color='red', linewidth=2.5, label='Current Fitted Line')
    
    # Scatter plot of actual data points on top
    ax.scatter(X, y, color='blue', label='Actual Data', zorder=5)
    
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Placement (LPA)")
    ax.set_title("Regression Fit: CGPA vs Placement")
    
    # Place legend outside if too many items
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def main():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    st.title("Linear Regression Visualizer 📈")
    
    # Setup tabs
    tab1, tab2 = st.tabs(["y = mx + b Line Explorer", "Simple Linear Regression — CGPA vs Placement"])
    
    # -----------------
    # TAB 1: Line Explorer
    # -----------------
    with tab1:
        st.header("Explore the Equation of a Line")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Parameters")
            # Sliders for m and b
            m = st.slider("Slope (m)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
            b = st.slider("Intercept (b)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
            
            # Educational explanation
            st.markdown("### Explanation")
            st.info(
                "- **Slope (m)** controls the steepness and direction of the line. A higher positive value makes it steeper going up, while a negative value makes it slope down.\n"
                "- **Intercept (b)** is where the line crosses the y-axis (when x=0). Changing it shifts the entire line up or down without changing its steepness."
            )
            
        with col2:
            # Display current equation
            st.subheader(f"Current Equation: **y = {m:.1f}x + {b:.1f}**")
            # Plot
            fig = plot_line_explorer(m, b)
            st.pyplot(fig)

    # -----------------
    # TAB 2: Simple Linear Regression
    # -----------------
    with tab2:
        st.header("Linear Regression: CGPA vs Placement (LPA)")
        
        # Educational sidebar/expander
        with st.expander("📖 Learn about Simple Linear Regression", expanded=False):
            st.markdown("""
            ### Key Concepts
            - **Dependent vs Independent Variable**: 
                - The **Independent Variable (x)** is what we use to make predictions (e.g., CGPA).
                - The **Dependent Variable (y)** is what we are trying to predict (e.g., Placement LPA).
            - **Fitting a Line**: Linear regression tries to draw the "best fit" straight line through the data points. It minimizes the error (the distance between actual points and the line) using "Least Squares".
            - **Interpretation**: 
                - **Intercept (b)**: Theoretical placement if CGPA was 0.
                - **Slope (m)**: For every 1 point increase in CGPA, placement package increases by *m* LPA.
            - **R² (R-squared)**: A metric from 0 to 1 indicating how well the line fits the data (1 means perfect fit).
            """)

        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Data Table")
            st.write("Edit, add, or delete rows:")
            
            # Default dataset with realistic dummy data
            default_data = pd.DataFrame({
                'CGPA': [6.5, 7.0, 7.2, 7.8, 8.0, 8.3, 8.7, 9.0, 9.2, 9.5],
                'Placement_LPA': [3.5, 4.0, 4.5, 5.5, 6.0, 7.0, 8.5, 10.0, 11.5, 14.0]
            })
            
            # Interactive data editor
            edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                # Train model button
                train_clicked = st.button("Train Model", type="primary")
            with col_btn2:
                if st.session_state['history']:
                    if st.button("Clear History"):
                        st.session_state['history'] = []
                        st.session_state.pop('model', None)
                        st.rerun()

        with col2:
            if train_clicked:
                # Clean missing data if any empty rows were added
                clean_df = edited_df.dropna()
                if len(clean_df) < 2:
                    st.error("Need at least 2 data points to train.")
                else:
                    # Train model
                    model, r2, mse = train_and_evaluate(clean_df)
                    
                    # Store model in session state for predictions later
                    st.session_state['model'] = model
                    st.session_state['current_df'] = clean_df
                    
                    # Save to history
                    run_num = len(st.session_state['history']) + 1
                    m_learned = model.coef_[0]
                    b_learned = model.intercept_
                    st.session_state['history'].append({
                        "Run #": run_num,
                        "m": m_learned,
                        "b": b_learned,
                        "Equation": f"y = {m_learned:.2f}x + {b_learned:.2f}",
                        "R²": round(r2, 4),
                        "MSE": round(mse, 4)
                    })
                    st.success(f"Run {run_num} trained successfully!")
                    
            if 'model' in st.session_state and len(st.session_state['history']) > 0:
                current_run = st.session_state['history'][-1]
                
                # Display metrics and equation
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                metrics_col1.metric("Equation", current_run["Equation"])
                metrics_col2.metric("R² Score", f"{current_run['R²']:.4f}")
                metrics_col3.metric("MSE", f"{current_run['MSE']:.4f}")
                
                # Plot the regression
                st.subheader("Regression Plot")
                
                # Overlay selection
                history_options = [h["Run #"] for h in st.session_state['history'][:-1]]
                selected_runs = st.multiselect("Overlay previous runs:", options=history_options, default=[])
                
                fig2 = plot_regression(st.session_state['current_df'], st.session_state['model'], st.session_state['history'], selected_runs)
                st.pyplot(fig2)
                
                # Display History Table
                st.subheader("Run History")
                history_df = pd.DataFrame(st.session_state['history'])
                st.dataframe(history_df[["Run #", "Equation", "R²", "MSE"]], use_container_width=True, hide_index=True)
            else:
                st.info("Click 'Train Model' to train the regression model on the data.")
                
        # Predict Section below the columns
        st.divider()
        st.subheader("Predict Placement")
        
        if 'model' in st.session_state:
            pred_cgpa = st.number_input("Enter CGPA to predict Placement (LPA):", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
            predicted_placement = st.session_state['model'].predict([[pred_cgpa]])[0]
            st.write(f"### Predicted Placement: **{predicted_placement:.2f} LPA**")
        else:
            st.warning("Train the model first to make predictions.")

if __name__ == "__main__":
    main()
