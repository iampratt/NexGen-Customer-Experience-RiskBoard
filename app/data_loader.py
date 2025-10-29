from pathlib import Path
import pandas as pd
import streamlit as st


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

    orders = _read_csv(data_dir / REQUIRED_FILES["orders"])  # Order_ID, Order_Date, ...
    delivery = _read_csv(data_dir / REQUIRED_FILES["delivery"])  # Promised vs Actual ...
    routes = _read_csv(data_dir / REQUIRED_FILES["routes"])  # Distance, Traffic, Weather
    feedback = _read_csv(data_dir / REQUIRED_FILES["feedback"])  # Rating, Text
    costs = _read_csv(data_dir / REQUIRED_FILES["costs"])  # Fuel, Labor, etc.

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


