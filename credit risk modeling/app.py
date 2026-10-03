from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "german_credit_data.csv"
MODEL_PATH = APP_DIR / "credit_risk_pipeline.joblib"
FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Purpose",
]


@st.cache_resource
def load_model_assets() -> Any:
    """Load the newly trained preprocessing and prediction pipeline."""
    model = joblib.load(MODEL_PATH)
    model_features = list(getattr(model, "feature_names_in_", []))

    if model_features != FEATURE_COLUMNS:
        raise ValueError(
            "The model's input columns do not match this app. "
            f"Expected {FEATURE_COLUMNS}; found {model_features or 'no feature names'}."
        )

    return model


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def prepare_features(values: pd.DataFrame) -> pd.DataFrame:
    """Validate incoming records and prepare raw values for the fitted pipeline."""
    missing = [column for column in FEATURE_COLUMNS if column not in values.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")

    features = values.loc[:, FEATURE_COLUMNS].copy()
    for column in ("Age", "Job", "Credit amount", "Duration"):
        features[column] = pd.to_numeric(features[column], errors="coerce")
        if features[column].isna().any():
            raise ValueError(f"'{column}' must contain valid numbers in every row.")

    if not features["Age"].between(18, 100).all():
        raise ValueError("Age must be between 18 and 100.")
    if not features["Job"].isin([0, 1, 2, 3]).all():
        raise ValueError("Job must be one of 0, 1, 2, or 3.")
    if not features["Credit amount"].between(250, 18424).all():
        raise ValueError("Credit amount must be between 250 and 18,424.")
    if not features["Duration"].between(1, 72).all():
        raise ValueError("Duration must be between 1 and 72 months.")

    for column, allowed_values in {
        "Sex": {"female", "male"},
        "Housing": {"free", "own", "rent"},
        "Saving accounts": {"little", "moderate", "quite rich", "rich"},
        "Checking account": {"little", "moderate", "rich"},
        "Purpose": {
            "business",
            "car",
            "domestic appliances",
            "education",
            "furniture/equipment",
            "radio/tv",
            "repairs",
            "vacation/others",
        },
    }.items():
        features[column] = features[column].map(
            lambda value: (
                str(value).strip().lower()
                if pd.notna(value)
                and str(value).strip().lower() not in {"not provided", "unknown"}
                else float("nan")
            )
        )
        unknown = sorted(set(features[column].dropna()) - allowed_values)
        if unknown:
            raise ValueError(
                f"Unsupported value(s) for '{column}': {', '.join(unknown)}. "
                f"Supported values: {', '.join(sorted(allowed_values))}."
            )

    return features


def score_records(
    values: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    features = prepare_features(values)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    result = values.copy()
    result["Predicted risk"] = predictions

    class_labels = model.classes_
    probability_by_label = {
        label: probabilities[:, index]
        for index, label in enumerate(class_labels)
    }
    for label, scores in probability_by_label.items():
        result[f"{label.title()} probability (%)"] = (scores * 100).round(1)

    return result


def render_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --ink: #182b36; --muted: #687b83; --teal: #087f78; --line: #e4eceb; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: #f5f8f7; color: var(--ink); }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; color: var(--ink); }
        .hero {
            background: linear-gradient(115deg, #102e38 0%, #075e62 62%, #138c7c 100%);
            padding: 2rem 2.2rem; border-radius: 20px; color: #f6fffc;
            margin-bottom: 1.5rem; box-shadow: 0 14px 36px rgba(18, 61, 67, .14);
        }
        .hero h1 { color: #fff !important; margin: .35rem 0 .45rem 0; font-size: 2.15rem; }
        .hero p { color: #d2e9e5; margin: 0; font-size: 1rem; }
        .eyebrow { color: #9cddd0; font-size: .74rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        div[data-testid="stMetric"] {
            background: #fff; border: 1px solid var(--line); border-radius: 14px;
            padding: 1rem 1.1rem; box-shadow: 0 4px 14px rgba(27, 54, 59, .035);
        }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--ink); font-family: 'Manrope', sans-serif; }
        .result-card {
            padding: 1.5rem; border-radius: 16px; background: #fff;
            border: 1px solid var(--line); min-height: 190px;
        }
        .result-good { border-top: 4px solid #149477; }
        .result-bad { border-top: 4px solid #d87558; }
        .result-title { color: var(--muted); font-weight: 600; font-size: .83rem; text-transform: uppercase; letter-spacing: .08em; }
        .result-value { font: 800 2rem 'Manrope', sans-serif; margin: .55rem 0; color: var(--ink); }
        .result-note { color: var(--muted); line-height: 1.55; }
        .small-note { color: var(--muted); font-size: .84rem; }
        div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
            background: #087f78; border: 0; border-radius: 10px; font-weight: 700; min-height: 2.8rem;
        }
        div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {
            background: #066a65; border: 0;
        }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_portfolio(dataset: pd.DataFrame) -> None:
    st.subheader("Portfolio snapshot")
    if "Risk" not in dataset.columns:
        st.info("The bundled dataset does not include a Risk column to summarize.")
        return

    risk_counts = dataset["Risk"].dropna().value_counts()
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Historical risk mix")
        st.bar_chart(risk_counts, color="#138c7c", height=260)
    with right:
        st.markdown("#### Dataset preview")
        st.dataframe(dataset.drop(columns=["Unnamed: 0"], errors="ignore").head(8), hide_index=True)
    st.caption(
        "These are descriptive statistics from the bundled historical dataset; "
        "they are not a measure of live portfolio performance."
    )


def render_batch_scoring(
    model: Any,
) -> None:
    st.subheader("Score a group of applications")
    st.write("Upload a CSV with one application per row and all nine model input columns.")
    st.code(", ".join(FEATURE_COLUMNS), language=None)
    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Column names must match the list above. Extra columns are kept in the results.",
        key="batch_csv",
    )
    if uploaded is None:
        st.caption("No file is uploaded. Download this template to see the required format.")
        template = pd.DataFrame(
            [
                {
                    "Age": 35,
                    "Sex": "female",
                    "Job": 2,
                    "Housing": "own",
                    "Saving accounts": "moderate",
                    "Checking account": "little",
                    "Credit amount": 2500,
                    "Duration": 18,
                    "Purpose": "car",
                }
            ]
        )
        st.download_button(
            "Download CSV template",
            template.to_csv(index=False).encode("utf-8"),
            file_name="credit_risk_template.csv",
            mime="text/csv",
        )
        return

    try:
        records = pd.read_csv(uploaded)
        if records.empty:
            st.error("The uploaded CSV has no application rows.")
            return
        result = score_records(records, model)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Unable to score this CSV: {exc}")
        return

    good_count = int((result["Predicted risk"] == "good").sum())
    bad_count = len(result) - good_count
    good_metric, bad_metric = st.columns(2)
    good_metric.metric("Predicted good risk", good_count)
    bad_metric.metric("Predicted bad risk", bad_count)
    st.dataframe(result, hide_index=True, use_container_width=True)
    st.download_button(
        "Download scored results",
        result.to_csv(index=False).encode("utf-8"),
        file_name="credit_risk_scored.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    st.set_page_config(
        page_title="CreditLens | Risk Relay",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_styles()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Risk Relay · Credit intelligence</div>
          <h1>Make every credit decision clearer.</h1>
          <p>Explore historical patterns and assess applications with the trained Extra Trees model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_model_assets()
        dataset = load_dataset()
    except (OSError, EOFError, ValueError, ImportError) as exc:
        st.error(f"Unable to start the app: {exc}")
        st.info(
            "Confirm that the trained model and dataset files are beside app.py, "
            "then install the packages in requirements.txt."
        )
        st.stop()

    risk_counts = dataset["Risk"].dropna().value_counts() if "Risk" in dataset else pd.Series(dtype=int)
    good_share = (
        f"{risk_counts.get('good', 0) / risk_counts.sum():.0%}"
        if risk_counts.sum()
        else "—"
    )
    st.metric("Good-risk share", good_share)

    overview_tab, assess_tab, batch_tab = st.tabs(
        ["Portfolio overview", "Assess an application", "Batch scoring"]
    )
    with overview_tab:
        render_portfolio(dataset)

    with assess_tab:
        st.subheader("Application profile")
        st.write("Enter the attributes used by the supplied model to get a risk estimate.")
        with st.form("application_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", min_value=18, max_value=100, value=35)
                sex = st.selectbox("Sex", options=["female", "male"])
                job = st.selectbox(
                    "Job category",
                    options=[0, 1, 2, 3],
                    format_func=lambda value: f"Category {value}",
                    index=2,
                    help="The dataset encodes occupation skill level from 0 to 3.",
                )
                credit_amount = st.number_input(
                    "Credit amount",
                    min_value=250,
                    max_value=18424,
                    value=2500,
                    step=50,
                )
            with col2:
                housing = st.selectbox("Housing", options=["free", "own", "rent"])
                savings = st.selectbox(
                    "Savings account",
                    options=["Not provided", "little", "moderate", "quite rich", "rich"],
                )
                checking = st.selectbox(
                    "Checking account",
                    options=["Not provided", "little", "moderate", "rich"],
                )
                duration = st.slider(
                    "Credit duration (months)",
                    min_value=1,
                    max_value=72,
                    value=18,
                )
                purpose = st.selectbox(
                    "Loan purpose",
                    options=[
                        "business",
                        "car",
                        "domestic appliances",
                        "education",
                        "furniture/equipment",
                        "radio/TV",
                        "repairs",
                        "vacation/others",
                    ],
                )
            submitted = st.form_submit_button(
                "Assess credit risk",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            values = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Sex": sex,
                        "Job": job,
                        "Housing": housing,
                        "Saving accounts": savings,
                        "Checking account": checking,
                        "Credit amount": credit_amount,
                        "Duration": duration,
                        "Purpose": purpose,
                    }
                ]
            )
            try:
                scored = score_records(values, model)
            except ValueError as exc:
                st.error(f"Unable to assess this application: {exc}")
            else:
                risk = str(scored.loc[0, "Predicted risk"])
                probability_column = f"{risk.title()} probability (%)"
                probability = float(
                    scored[probability_column].to_numpy(dtype="float64")[0]
                )
                risk_class = "good" if risk == "good" else "bad"
                description = (
                    "The model places this profile in the good-risk class."
                    if risk == "good"
                    else "The model places this profile in the bad-risk class."
                )
                st.markdown(
                    f"""
                    <div class="result-card result-{risk_class}">
                      <div class="result-title">Model estimate</div>
                      <div class="result-value">{risk.title()} risk</div>
                      <div class="result-note">{description}<br>
                      Class probability: <strong>{probability:.1f}%</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(max(probability / 100, 0.0), 1.0))
                st.warning(
                    "This estimate is for educational purposes only. It is not a lending "
                    "decision, financial advice, or a substitute for fair-lending review "
                    "and human assessment."
                )

    with batch_tab:
        render_batch_scoring(model)

    st.divider()
    st.markdown(
        '<p class="small-note">Risk Relay is a demonstration application. '
        "Model outputs can be inaccurate and should not be used as the sole basis "
        "for a real credit decision.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()