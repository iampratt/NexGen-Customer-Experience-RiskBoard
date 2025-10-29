import numpy as np
import pandas as pd
import streamlit as st


def render_segment(df: pd.DataFrame) -> None:
    st.subheader("Segment Drilldown")
    segs = sorted(df["Customer_Segment"].dropna().unique())
    seg = st.selectbox("Segment", segs) if segs else None
    if seg is None:
        st.info("No segment data available.")
        return

    sdf = df[df["Customer_Segment"] == seg].copy()
    st.metric("Avg risk", f"{sdf['risk_blend'].mean():.1f}")
    st.metric("High-risk share", f"{(sdf['risk_blend']>65).mean()*100:.1f}%")

    # Priority x Risk matrix
    if "Priority" in sdf.columns:
        pivot = sdf.pivot_table(index=["Priority", "Carrier"], values="risk_blend", aggfunc=["mean", "count"]).round(1)
        st.dataframe(pivot)

    # Driver hints (coef from trained LR if stored)
    st.markdown("**Top drivers (heuristic proxies)**")
    drivers = {
        "Delays": sdf["delay_days"].mean(),
        "Low ratings": (sdf["Customer_Rating"] <= 2).mean(),
        "Quality issues": sdf["Quality_Issue"].isin(["Wrong_Item","Minor_Damage","Major_Damage"]).mean(),
        "Traffic": sdf["Traffic_Delay_Minutes"].mean(),
        "Negative text": sdf["neg_sentiment"].mean(),
    }
    st.json({k: float(v) if not isinstance(v, (int, float)) else v for k, v in drivers.items()})

    # Suggested interventions
    st.markdown("**Interventions**")
    st.write("- Proactive ETA updates for delayed orders")
    st.write("- Priority handling for Express with high risk")
    st.write("- QA audit for carriers with elevated issues")


