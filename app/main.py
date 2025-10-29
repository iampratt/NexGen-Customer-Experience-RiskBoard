import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path when running via `streamlit run app/main.py`
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import load_all_data
from app.features import build_dataset, compute_heuristic_risk
from app.model import get_or_train_model, blend_risk_scores
from app.views.overview import render_overview
from app.views.segment import render_segment
from app.views.orders import render_orders


st.set_page_config(
    page_title="NexGen CX Risk Dashboard",
    page_icon="📦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _load_joined_df(data_dir: Path) -> pd.DataFrame:
    data = load_all_data(data_dir)
    df = build_dataset(**data)
    return df


def main() -> None:
    st.title("NexGen Logistics • Customer Experience Risk Dashboard")

    # Data location is always the project root (CSV files reside here)
    data_dir_path = Path(os.getcwd())

    df = _load_joined_df(data_dir_path)

    # Derive global filter options
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

    # Filter df
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

    # Scores
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

    # Downloads
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


