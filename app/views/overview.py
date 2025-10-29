import numpy as np
import pandas as pd
import streamlit as st

from ..components.charts import bar_by_segment, line_trend, scatter_cost_delay, issues_distribution
from ..model import get_model_metrics, get_coefficient_importances


def _risk_band(score: float) -> str:
    if score > 65:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")
    st.caption("Hybrid risk = 50% heuristic (delays, quality, cost, route, sentiment) + 50% ML (logistic regression)")
    df = df.copy()
    df["risk_band"] = df["risk_blend"].apply(_risk_band)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("High-risk orders", int((df["risk_band"] == "High").sum()))
    with col2:
        st.metric("Avg delay (days)", f"{df['delay_days'].mean():.1f}")
    with col3:
        st.metric("Avg CSAT", f"{df['Customer_Rating'].mean():.2f}")
    with col4:
        st.metric("Orders", len(df))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_by_segment(df), use_container_width=True)
    with c2:
        st.plotly_chart(issues_distribution(df), use_container_width=True)

    st.plotly_chart(line_trend(df), use_container_width=True)
    st.plotly_chart(scatter_cost_delay(df), use_container_width=True)

    st.markdown("---")
    st.markdown("**Model metrics (recent test split)**")
    # Access model from session if available via caller
    if "_model_ctx" in st.session_state:
        model, feature_cols = st.session_state["_model_ctx"]
        metrics = get_model_metrics(df, model, feature_cols)
        st.json(metrics)

        st.markdown("**Top coefficients**")
        imp = get_coefficient_importances(model, feature_cols)
        if not imp.empty:
            st.dataframe(imp.head(15))
        else:
            st.info("Importances unavailable for current model.")

    st.markdown("---")
    st.markdown("**Business Impact Calculator**")
    with st.expander("Estimate ROI from proactive CX interventions", expanded=False):
        colA, colB, colC = st.columns(3)
        with colA:
            high_risk = int((df.get("risk_blend", 0) > 65).sum())
            contacts = st.number_input("Proactive contacts per week", value=float(min(50, max(10, high_risk))), step=5.0)
            save_rate = st.slider("Save rate of contacted high-risk (%)", min_value=0, max_value=100, value=25)
        with colB:
            avg_order_value = st.number_input("Avg order value (INR)", value=1500.0, step=100.0)
            credit_cost = st.number_input("Avg credit/compensation per save (INR)", value=150.0, step=10.0)
        with colC:
            weeks = st.number_input("Horizon (weeks)", value=4.0, step=1.0)

        saves = contacts * (save_rate / 100.0) * weeks
        gross = saves * avg_order_value
        cost = saves * credit_cost
        net = gross - cost

        st.write(f"Estimated saves: {int(saves)}")
        st.write(f"Gross retained revenue: INR {gross:,.0f}")
        st.write(f"Credits cost: INR {cost:,.0f}")
        st.success(f"Net benefit over {int(weeks)} weeks: INR {net:,.0f}")


