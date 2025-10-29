from typing import List
import numpy as np
import pandas as pd


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
    return min(score, 2.0) / 2.0  # normalize to 0..1


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
    # feedback: many-to-one (latest feedback per order)
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
    # Normalize helper
    def _norm(col: pd.Series) -> pd.Series:
        col = col.astype(float)
        rng = col.max() - col.min()
        if rng == 0 or np.isnan(rng):
            return pd.Series(0.0, index=col.index)
        return (col - col.min()) / rng

    delay_score = _norm(df.get("delay_days", pd.Series(0, index=df.index)))
    rating_raw = df.get("Customer_Rating", pd.Series(np.nan, index=df.index)).fillna(3)
    rating_score = 1 - (rating_raw.clip(1, 5) - 1) / 4.0  # low rating -> high risk
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
        # categorical (will be one-hot in pipeline):
        "Delivery_Status",
        "Quality_Issue",
        "Weather_Impact",
        "Customer_Segment",
        "Priority",
        "Product_Category",
        "Carrier",
    ]
    return [c for c in base_cols if c in df.columns]


