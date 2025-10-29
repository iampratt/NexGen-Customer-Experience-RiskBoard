import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from typing import List, Tuple, Dict


REQUIRED_FILES = {
    "orders": "orders.csv",
    "delivery": "delivery_performance.csv",
    "routes": "routes_distance.csv",
    "feedback": "customer_feedback.csv",
    "costs": "cost_breakdown.csv",
}

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data(show_spinner=False)
def load_all_data(data_dir: Path) -> dict:
    missing = [name for name, fname in REQUIRED_FILES.items() if not (data_dir / fname).exists()]
    if missing:
        st.error(f"Missing required files in {data_dir}: {missing}")
        st.stop()

    orders = _read_csv(data_dir / REQUIRED_FILES["orders"])
    delivery = _read_csv(data_dir / REQUIRED_FILES["delivery"])
    routes = _read_csv(data_dir / REQUIRED_FILES["routes"])
    feedback = _read_csv(data_dir / REQUIRED_FILES["feedback"])
    costs = _read_csv(data_dir / REQUIRED_FILES["costs"])

    # Normalize dtypes
    orders["Order_Date"] = pd.to_datetime(orders["Order_Date"], errors="coerce")
    if "Feedback_Date" in feedback.columns:
        feedback["Feedback_Date"] = pd.to_datetime(feedback["Feedback_Date"], errors="coerce")

    return {
        "orders": orders,
        "delivery": delivery,
        "routes": routes,
        "feedback": feedback,
        "costs": costs,
    }

NEGATIVE_LEXICON = {
    "delayed": 1.0,
    "wrong": 0.8,
    "damage": 0.7,
    "poor": 0.6,
    "late": 0.6,
    "not acceptable": 1.0,
    "bad": 0.6,
    "refund": 0.7,
}

def _sentiment_score(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    t = text.lower()
    score = 0.0
    for k, w in NEGATIVE_LEXICON.items():
        if k in t:
            score += w
    return min(score, 2.0) / 2.0

def build_dataset(
    orders: pd.DataFrame,
    delivery: pd.DataFrame,
    routes: pd.DataFrame,
    feedback: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    df = orders.merge(delivery, on="Order_ID", how="left")
    df = df.merge(routes, on="Order_ID", how="left")
    df = df.merge(costs, on="Order_ID", how="left")
    feedback_sorted = feedback.sort_values(["Order_ID", "Feedback_Date"]).drop_duplicates("Order_ID", keep="last")
    df = df.merge(feedback_sorted, on="Order_ID", how="left")

    # Basic features
    df["Promised_Delivery_Days"] = pd.to_numeric(df["Promised_Delivery_Days"], errors="coerce")
    df["Actual_Delivery_Days"] = pd.to_numeric(df["Actual_Delivery_Days"], errors="coerce")
    df["delay_days"] = (df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]).clip(lower=0)

    df["Delivery_Cost_INR"] = pd.to_numeric(df["Delivery_Cost_INR"], errors="coerce")
    df["Order_Value_INR"] = pd.to_numeric(df["Order_Value_INR"], errors="coerce")
    df["cost_ratio"] = (df["Delivery_Cost_INR"] / df["Order_Value_INR"]).replace([np.inf, -np.inf], np.nan)

    # Route stress
    df["Traffic_Delay_Minutes"] = pd.to_numeric(df["Traffic_Delay_Minutes"], errors="coerce")

    # Ratings
    if "Customer_Rating" in df.columns:
        df["Customer_Rating"] = pd.to_numeric(df["Customer_Rating"], errors="coerce")
    if "Rating" in df.columns:
        df["Feedback_Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    else:
        df["Feedback_Rating"] = np.nan

    # Sentiment from text
    df["neg_sentiment"] = df["Feedback_Text"].apply(_sentiment_score)

    # Missing indicators
    for col in [
        "delay_days",
        "Delivery_Cost_INR",
        "Traffic_Delay_Minutes",
        "Feedback_Rating",
        "cost_ratio",
        "Customer_Rating",
    ]:
        df[f"missing__{col}"] = df[col].isna().astype(int)
        if df[col].dtype.kind in {"f", "i"}:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    # Coerce categories
    for c in [
        "Delivery_Status",
        "Quality_Issue",
        "Weather_Impact",
        "Customer_Segment",
        "Priority",
        "Product_Category",
        "Carrier",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    return df

def compute_heuristic_risk(df: pd.DataFrame) -> pd.Series:
    def _norm(col: pd.Series) -> pd.Series:
        col = col.astype(float)
        rng = col.max() - col.min()
        if rng == 0 or np.isnan(rng):
            return pd.Series(0.0, index=col.index)
        return (col - col.min()) / rng

    delay_score = _norm(df.get("delay_days", pd.Series(0, index=df.index)))
    rating_raw = df.get("Customer_Rating", pd.Series(np.nan, index=df.index)).fillna(3)
    rating_score = 1 - (rating_raw.clip(1, 5) - 1) / 4.0
    quality_issue = df.get("Quality_Issue", pd.Series("Perfect", index=df.index)).astype(str)
    quality_score = quality_issue.isin(["Wrong_Item", "Minor_Damage", "Major_Damage"]).astype(float)
    cost_score = _norm(df.get("cost_ratio", pd.Series(0, index=df.index)).fillna(0))
    route_score = _norm(df.get("Traffic_Delay_Minutes", pd.Series(0, index=df.index))) + (
        df.get("Weather_Impact", pd.Series("None", index=df.index)).astype(str).isin(["Fog", "Heavy_Rain"]).astype(float) * 0.5
    )
    route_score = route_score.clip(0, 1)
    text_score = df.get("neg_sentiment", pd.Series(0, index=df.index))

    risk = (
        0.40 * delay_score
        + 0.30 * ((rating_score + text_score) / 2.0)
        + 0.15 * quality_score
        + 0.10 * cost_score
        + 0.05 * route_score
    )
    return (100 * risk).clip(0, 100)

def label_at_risk(df: pd.DataFrame) -> pd.Series:
    severe_delay = df.get("Delivery_Status", pd.Series("", index=df.index)).astype(str).eq("Severely-Delayed")
    low_rating = df.get("Customer_Rating", pd.Series(np.nan, index=df.index)).fillna(3) <= 2
    issue = df.get("Quality_Issue", pd.Series("", index=df.index)).astype(str).isin(["Wrong_Item", "Minor_Damage", "Major_Damage"])
    return (severe_delay | low_rating | issue).astype(int)

def model_feature_columns(df: pd.DataFrame) -> List[str]:
    base_cols = [
        "delay_days",
        "Customer_Rating",
        "Feedback_Rating",
        "neg_sentiment",
        "cost_ratio",
        "Traffic_Delay_Minutes",
        "Delivery_Status",
        "Quality_Issue",
        "Weather_Impact",
        "Customer_Segment",
        "Priority",
        "Product_Category",
        "Carrier",
    ]
    return [c for c in base_cols if c in df.columns]

# Original: model.py
MODEL_PATH = Path("models/cx_lr.joblib")

@st.cache_resource(show_spinner=False)
def get_or_train_model(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    feature_cols = model_feature_columns(df)
    X = df[feature_cols].copy()
    y = label_at_risk(df)

    numeric_cols = [c for c in feature_cols if X[c].dtype.kind in {"f", "i"}]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    model = Pipeline(steps=[("pre", pre), ("clf", clf)])

    if "Order_Date" in df.columns:
        df_sorted = df.sort_values("Order_Date").reset_index(drop=True)
        split_idx = int(len(df_sorted) * 0.8)
        train_idx = df_sorted.index[:split_idx]
        test_idx = df_sorted.index[split_idx:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
    except Exception:
        pass

    return model, feature_cols

def blend_risk_scores(df: pd.DataFrame, model: Pipeline, feature_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    X = df[feature_cols].copy()
    try:
        prob = model.predict_proba(X)[:, 1]
    except Exception:
        prob = np.zeros(len(df))
    risk_ml = 100.0 * prob
    if "risk_heuristic" in df.columns:
        blend = 0.5 * df["risk_heuristic"].astype(float).values + 0.5 * risk_ml
    else:
        blend = risk_ml
    return pd.Series(blend).clip(0, 100), pd.Series(prob)

def get_model_metrics(df: pd.DataFrame, model: Pipeline, feature_cols: List[str]) -> Dict[str, float]:
    y_true = label_at_risk(df)
    X = df[feature_cols].copy()
    if "Order_Date" in df.columns:
        order_idx = df.sort_values("Order_Date").index
        X_sorted = X.loc[order_idx]
        y_sorted = y_true.loc[order_idx].values
        split_idx = int(len(df) * 0.8)
        X_test = X_sorted.iloc[split_idx:]
        y = y_sorted[split_idx:]
    else:
        split_idx = int(len(df) * 0.8)
        X_test = X.iloc[split_idx:]
        y = y_true.iloc[split_idx:].values

    try:
        prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        prob = np.zeros(len(y))
    roc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")

    y_pred = (prob >= 0.65).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    return {"roc_auc": roc, "precision@0.65": float(p), "recall@0.65": float(r), "f1@0.65": float(f1)}

def get_coefficient_importances(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    try:
        pre = model.named_steps.get("pre")
        clf = model.named_steps.get("clf")
        if pre is None or clf is None or not hasattr(clf, "coef_"):
            return pd.DataFrame(columns=["feature", "weight"])
        names = pre.get_feature_names_out(feature_cols)
        coefs = clf.coef_.ravel()
        imp = pd.DataFrame({"feature": names, "weight": coefs}).sort_values("weight", ascending=False)
        return imp
    except Exception:
        return pd.DataFrame(columns=["feature", "weight"])

# Original: charts.py
def bar_by_segment(df: pd.DataFrame, value_col: str = "risk_blend"):
    agg = df.groupby("Customer_Segment", dropna=False)[value_col].mean().reset_index()
    fig = px.bar(agg, x="Customer_Segment", y=value_col, color="Customer_Segment", title="Avg Risk by Segment")
    fig.update_layout(showlegend=False)
    return fig

def line_trend(df: pd.DataFrame):
    d = df.copy()
    d["Order_Week"] = pd.to_datetime(d["Order_Date"]).dt.to_period("W").dt.start_time
    agg = d.groupby("Order_Week").agg(avg_delay=("delay_days", "mean"), csat=("Customer_Rating", "mean")).reset_index()
    fig = px.line(agg, x="Order_Week", y=["avg_delay", "csat"], title="Weekly Delay and CSAT Trend")
    return fig

def scatter_cost_delay(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="delay_days",
        y="Delivery_Cost_INR",
        color="risk_blend",
        hover_data=["Order_ID", "Customer_Segment", "Priority"],
        title="Cost vs Delay colored by Risk",
        color_continuous_scale="RdYlGn_r",
    )
    return fig

def issues_distribution(df: pd.DataFrame):
    counts = df["Quality_Issue"].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["Quality_Issue", "count"]
    fig = px.bar(counts, x="Quality_Issue", y="count", title="Quality Issues Distribution")
    return fig

# Original: orders.py
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
    status = str(row.get("Delivery_Status", ""))
    issue = str(row.get("Quality_Issue", ""))
    priority = str(row.get("Priority", ""))
    delay = float(row.get("delay_days", 0) or 0)

    if issue in {"Wrong_Item", "Minor_Damage", "Major_Damage"}:
        actions.append("Offer compensation credit and dispatch priority re-ship")
    if status == "Severely-Delayed" or delay >= 2:
        actions.append("Proactive outreach with revised ETA and ops escalation")
    if status == "Slightly-Delayed" or (delay >= 1 and delay < 2):
        actions.append("Send ETA update and monitor carrier performance")
    if priority == "Express" and delay > 0:
        actions.append("Upgrade route and assign top-performing carrier")

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

    high = df[df["risk_blend"] > 65]
    if not high.empty:
        st.download_button(
            label="Download high-risk orders (CSV)",
            data=high.to_csv(index=False).encode("utf-8"),
            file_name="high_risk_orders.csv",
            mime="text/csv",
        )

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

    if "Priority" in sdf.columns:
        pivot = sdf.pivot_table(index=["Priority", "Carrier"], values="risk_blend", aggfunc=["mean", "count"]).round(1)
        st.dataframe(pivot)

    st.markdown("**Top drivers (heuristic proxies)**")
    drivers = {
        "Delays": sdf["delay_days"].mean(),
        "Low ratings": (sdf["Customer_Rating"] <= 2).mean(),
        "Quality issues": sdf["Quality_Issue"].isin(["Wrong_Item","Minor_Damage","Major_Damage"]).mean(),
        "Traffic": sdf["Traffic_Delay_Minutes"].mean(),
        "Negative text": sdf["neg_sentiment"].mean(),
    }
    st.json({k: float(v) if not isinstance(v, (int, float)) else v for k, v in drivers.items()})

    st.markdown("**Interventions**")
    st.write("- Proactive ETA updates for delayed orders")
    st.write("- Priority handling for Express with high risk")
    st.write("- QA audit for carriers with elevated issues")

# Main application
@st.cache_data(show_spinner=False)
def _load_joined_df(data_dir: Path) -> pd.DataFrame:
    data = load_all_data(data_dir)
    df = build_dataset(**data)
    return df

def main() -> None:
    st.set_page_config(
        page_title="NexGen CX Risk Dashboard",
        page_icon="📦",
        layout="wide",
    )

    st.title("NexGen Logistics • Customer Experience Risk Dashboard")

    data_dir_path = Path(os.getcwd())
    df = _load_joined_df(data_dir_path)

    min_date = pd.to_datetime(df["Order_Date"]).min()
    max_date = pd.to_datetime(df["Order_Date"]).max()
    start_date, end_date = st.sidebar.date_input(
        "Order date range",
        value=(min_date.to_pydatetime().date(), max_date.to_pydatetime().date()),
    )

    segment_opts = ["All"] + sorted([x for x in df["Customer_Segment"].dropna().unique()])
    segment = st.sidebar.selectbox("Customer segment", segment_opts)

    priority_opts = ["All"] + sorted([x for x in df["Priority"].dropna().unique()])
    priority = st.sidebar.selectbox("Priority", priority_opts)

    product_opts = ["All"] + sorted([x for x in df["Product_Category"].dropna().unique()])
    product = st.sidebar.selectbox("Product category", product_opts)

    include_in_transit = st.sidebar.toggle("Include in-transit orders", value=True)

    fdf = df.copy()
    fdf = fdf[(pd.to_datetime(fdf["Order_Date"]).dt.date >= start_date) & (pd.to_datetime(fdf["Order_Date"]).dt.date <= end_date)]
    if segment != "All":
        fdf = fdf[fdf["Customer_Segment"] == segment]
    if priority != "All":
        fdf = fdf[fdf["Priority"] == priority]
    if product != "All":
        fdf = fdf[fdf["Product_Category"] == product]
    if not include_in_transit:
        fdf = fdf[~fdf["Delivery_Status"].isin(["In-Transit"])].copy()

    fdf["risk_heuristic"] = compute_heuristic_risk(fdf)
    model, feature_cols = get_or_train_model(df)
    st.session_state["_model_ctx"] = (model, feature_cols)
    fdf["risk_blend"], fdf["risk_ml_prob"] = blend_risk_scores(fdf, model, feature_cols)

    view = st.radio(
        "View",
        options=["Overview", "Segment", "Orders"],
        horizontal=True,
    )

    if view == "Overview":
        render_overview(fdf)
    elif view == "Segment":
        render_segment(fdf)
    else:
        render_orders(fdf)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Export")
    csv_bytes = fdf.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Download filtered data (CSV)",
        data=csv_bytes,
        file_name="filtered_orders_with_risk.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()