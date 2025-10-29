import pandas as pd
import streamlit as st


def _risk_badge(score: float) -> str:
    if score > 65:
        return "🟥 High"
    if score >= 35:
        return "🟨 Medium"
    return "🟩 Low"


def _suggest_actions(row: pd.Series) -> list:
    risk_val = row.get("risk_blend", 0)
    try:
        risk = float(risk_val) if pd.notna(risk_val) else 0.0
    except Exception:
        risk = 0.0

    actions = []
    # Context signals
    status = str(row.get("Delivery_Status", ""))
    issue = str(row.get("Quality_Issue", ""))
    priority = str(row.get("Priority", ""))
    delay = float(row.get("delay_days", 0) or 0)

    # Always act on critical service failures regardless of risk score
    if issue in {"Wrong_Item", "Minor_Damage", "Major_Damage"}:
        actions.append("Offer compensation credit and dispatch priority re-ship")
    if status == "Severely-Delayed" or delay >= 2:
        actions.append("Proactive outreach with revised ETA and ops escalation")
    if status == "Slightly-Delayed" or (delay >= 1 and delay < 2):
        actions.append("Send ETA update and monitor carrier performance")
    if priority == "Express" and delay > 0:
        actions.append("Upgrade route and assign top-performing carrier")

    # Risk-driven generic actions
    if risk > 65:
        actions.append("Add credit-on-account and post-delivery CSAT follow-up")
    elif risk >= 35:
        actions.append("Enable proactive notifications until delivery completes")

    if not actions:
        actions.append("No action required")
    return actions


def render_orders(df: pd.DataFrame) -> None:
    st.subheader("Orders")
    st.caption("Risk bands: 🟥 High (>65), 🟨 Medium (35-65), 🟩 Low (<35)")
    cols = [
        "Order_ID",
        "Order_Date",
        "Customer_Segment",
        "Priority",
        "Product_Category",
        "Carrier",
        "Delivery_Status",
        "delay_days",
        "Customer_Rating",
        "risk_heuristic",
        "risk_ml_prob",
        "risk_blend",
    ]
    cols = [c for c in cols if c in df.columns]

    table = df[cols].copy()
    table = table.sort_values("risk_blend", ascending=False)
    table["Risk"] = table["risk_blend"].apply(_risk_badge)
    st.dataframe(table, use_container_width=True)

    # Bulk export high-risk orders
    high = df[df["risk_blend"] > 65]
    if not high.empty:
        st.download_button(
            label="Download high-risk orders (CSV)",
            data=high.to_csv(index=False).encode("utf-8"),
            file_name="high_risk_orders.csv",
            mime="text/csv",
        )

    # Detail drawer
    st.markdown("### Order details")
    oid = st.selectbox("Select Order_ID", df["Order_ID"].unique()) if not df.empty else None
    if oid is None:
        return
    od = df[df["Order_ID"] == oid].iloc[0]
    st.json({
        "Order_ID": od.get("Order_ID"),
        "Route": od.get("Route"),
        "Distance_KM": float(od.get("Distance_KM", 0)) if pd.notna(od.get("Distance_KM")) else None,
        "Delivery_Status": od.get("Delivery_Status"),
        "Quality_Issue": od.get("Quality_Issue"),
        "Risk": float(od.get("risk_blend", 0)),
        "Heuristic": float(od.get("risk_heuristic", 0)),
        "ML_Prob": float(od.get("risk_ml_prob", 0)),
    })
    st.write("Suggested action:")
    for a in _suggest_actions(od):
        st.markdown(f"- {a}")


