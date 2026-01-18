"""
Streamlit UI for the Student Performance Predictor.

Responsibilities:
- Collect student inputs
- Load trained model and feature columns
- Align inputs to training schema and predict final grade (G3)
- Categorize performance and serve recommendations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import APP_DESCRIPTION, APP_TITLE
from src.recommendation import (
    PerformanceLevel,
    RecommendationEngine,
    categorize_performance,
)

# -----------------------------------------------------------------------------
# Paths (deployment-safe)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root

MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.json"


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained RandomForestRegressor from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_feature_columns() -> List[str]:
    """Load ordered feature columns used during training for alignment."""
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature schema not found at: {FEATURE_COLUMNS_PATH}")

    with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def preprocess_user_input(user_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """One-hot encode user inputs and align to training columns."""
    cat_cols = user_df.select_dtypes(exclude="number").columns
    encoded = pd.get_dummies(user_df, columns=cat_cols, drop_first=True)
    aligned = encoded.reindex(columns=feature_columns, fill_value=0)
    return aligned


# -----------------------------------------------------------------------------
# UI Form
# -----------------------------------------------------------------------------
def build_input_form() -> Dict[str, Any]:
    """Collect minimal, high-impact student inputs for prediction."""
    st.subheader("Student Academic & Support Details")

    col1, col2 = st.columns(2)

    with col1:
        G1 = st.number_input("G1 (1st Period Grade)", 0, 20, 10)
        G2 = st.number_input("G2 (2nd Period Grade)", 0, 20, 10)
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], index=1)
        failures = st.selectbox("Past Failures", [0, 1, 2, 3], index=0)
        absences = st.number_input("Absences", 0, 93, 2)

    with col2:
        famsup = st.selectbox("Family Support", ["yes", "no"], index=1)
        schoolsup = st.selectbox("School Support", ["yes", "no"], index=1)
        higher = st.selectbox("Wants Higher Education", ["yes", "no"], index=0)
        internet = st.selectbox("Internet Access", ["yes", "no"], index=0)

    payload = {
        "G1": G1,
        "G2": G2,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "famsup": famsup,
        "schoolsup": schoolsup,
        "higher": higher,
        "internet": internet,
    }

    # Hidden defaults (ensure these match training features)
    payload.update({
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "traveltime": 1,
        "activities": "no",
        "romantic": "no",
        "famrel": 3,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 2,
        "health": 3,
    })

    return payload


# -----------------------------------------------------------------------------
# Output UI
# -----------------------------------------------------------------------------
def render_recommendations(score: float, level: PerformanceLevel, engine: RecommendationEngine):
    """Display predicted score, performance, and learning resources."""
    recs = engine.get_recommendations(score)

    st.success(f"Predicted Final Grade (G3): {score:.2f}")
    st.info(f"Performance Level: {level.value}")
    st.write(recs["advice"])

    st.subheader("Recommended Learning Resources")
    for item in recs["resources"]:
        title = item.get("title", "Resource")
        rtype = item.get("type", "N/A")
        platform = item.get("platform", "N/A")
        difficulty = item.get("difficulty", "N/A")
        link = item.get("link", "")

        if link:
            st.markdown(f"- **{title}** ({rtype}, {platform}) — {difficulty}  \n  🔗 {link}")
        else:
            st.markdown(f"- **{title}** ({rtype}, {platform}) — {difficulty}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    try:
        model = load_model()
        feature_columns = load_feature_columns()
    except Exception as exc:
        st.error(f"Failed to load model or feature schema: {exc}")
        st.code(f"MODEL_PATH = {MODEL_PATH}\nFEATURE_COLUMNS_PATH = {FEATURE_COLUMNS_PATH}")
        return

    engine = RecommendationEngine()

    with st.form("prediction_form"):
        user_inputs = build_input_form()
        submitted = st.form_submit_button("Predict Performance")

    if not submitted:
        return

    try:
        user_df = pd.DataFrame([user_inputs])
        aligned = preprocess_user_input(user_df, feature_columns)
        prediction = float(model.predict(aligned)[0])
        level = categorize_performance(prediction)
        render_recommendations(prediction, level, engine)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
