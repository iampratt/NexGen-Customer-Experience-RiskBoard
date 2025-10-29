from pathlib import Path
from typing import List, Tuple, Dict
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from .features import label_at_risk, model_feature_columns


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

    # Time-aware split surrogate: use Order_Date if present, else random
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

    # Persist model
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
    """Compute simple metrics on the most recent 20% as a proxy test split."""
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

    # Precision/Recall at high-risk threshold (0.65)
    y_pred = (prob >= 0.65).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    return {"roc_auc": roc, "precision@0.65": float(p), "recall@0.65": float(r), "f1@0.65": float(f1)}


def get_coefficient_importances(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    """Return sorted feature importances derived from logistic regression coefficients."""
    try:
        pre = model.named_steps.get("pre")
        clf = model.named_steps.get("clf")
        if pre is None or clf is None or not hasattr(clf, "coef_"):
            return pd.DataFrame(columns=["feature", "weight"])  # empty
        names = pre.get_feature_names_out(feature_cols)
        coefs = clf.coef_.ravel()
        imp = pd.DataFrame({"feature": names, "weight": coefs}).sort_values("weight", ascending=False)
        return imp
    except Exception:
        return pd.DataFrame(columns=["feature", "weight"])  # empty


