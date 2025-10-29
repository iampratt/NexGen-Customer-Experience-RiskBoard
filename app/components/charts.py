import pandas as pd
import plotly.express as px


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


