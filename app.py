
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ShopSense | Customer Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DARK NAVY + PINK/PURPLE THEME
# ============================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
    :root {
        --pink: #ff4fa3;
        --pink-dark: #ec4899;
        --purple: #a78bfa;
        --peach: #ffadad;
        --bg: #090b18;
        --surface: #15182b;
        --surface-light: #20243b;
        --text: #f8fafc;
        --muted: #b4b8cc;
        --line: #343751;
        --white: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }

    /* APP BACKGROUND */

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                ellipse at 4% 0%,
                rgba(168, 85, 247, 0.22),
                transparent 36%
            ),
            radial-gradient(
                ellipse at 96% 10%,
                rgba(236, 72, 153, 0.18),
                transparent 34%
            ),
            radial-gradient(
                ellipse at 55% 100%,
                rgba(99, 102, 241, 0.12),
                transparent 42%
            ),
            linear-gradient(
                135deg,
                #080a16 0%,
                #101329 48%,
                #171024 100%
            );
        background-attachment: fixed;
    }

    [data-testid="stMain"] {
        background: transparent;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 10, 22, 0.55);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* TEXT */

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: #f8fafc !important;
        letter-spacing: -0.035em;
    }

    p, li, label {
        color: #e2e5f2;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #e2e5f2;
    }

    [data-testid="stCaptionContainer"] {
        color: #b4b8cc;
    }

    /* BRAND */

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 5px;
    }

    .brand-mark {
        width: 46px;
        height: 46px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(135deg, #ec4899, #8b5cf6);
        box-shadow: 0 0 24px rgba(236, 72, 153, 0.30);
    }

    .brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: #ffffff;
    }

    .brand-caption {
        font-size: 0.78rem;
        color: #b4b8cc;
        margin-top: -3px;
    }

    /* HORIZONTAL NAVIGATION */

    div[data-testid="stSegmentedControl"] {
        width: 100%;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    div[data-testid="stSegmentedControl"] > label {
        display: none !important;
    }

    div[data-testid="stSegmentedControl"] [role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        align-items: center;
        gap: 8px;
        width: 100%;
        background: transparent !important;
        border: none !important;
    }

    div[data-testid="stSegmentedControl"] label {
        flex: 0 0 auto !important;
        background: #191c31 !important;
        color: #e9eaf5 !important;
        border: 1px solid #373a57 !important;
        border-radius: 12px !important;
        padding: 9px 16px !important;
        min-height: 39px;
        opacity: 1 !important;
        transition: all 0.18s ease;
    }

    div[data-testid="stSegmentedControl"] label:hover {
        background: #30203d !important;
        border-color: #ec4899 !important;
        color: #ffffff !important;
    }

    div[data-testid="stSegmentedControl"] label p,
    div[data-testid="stSegmentedControl"] label span {
        color: inherit !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    div[data-testid="stSegmentedControl"] label:has(input:checked) {
        background: linear-gradient(110deg, #ec4899, #8b5cf6) !important;
        border-color: transparent !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(236, 72, 153, 0.25);
    }

    div[data-testid="stSegmentedControl"] input {
        display: none !important;
    }

    /* PAGE HEADERS */

    .eyebrow {
        color: #ff78b7;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 25px;
    }

    .page-title {
        font-family: 'Manrope', sans-serif;
        color: #ffffff;
        font-size: clamp(1.9rem, 3vw, 2.75rem);
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.055em;
        margin: 8px 0;
    }

    .page-subtitle {
        color: #c0c4d8;
        max-width: 780px;
        font-size: 0.95rem;
        margin-bottom: 22px;
        line-height: 1.65;
    }

    /* HERO */

    .hero {
        background: linear-gradient(
            115deg,
            #ec4899 0%,
            #c044c7 50%,
            #7652e8 100%
        );
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 25px;
        padding: 32px;
        color: white;
        margin: 22px 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.30);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -65px;
        top: -120px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.45);
        box-shadow:
            0 0 0 28px rgba(255,255,255,.09),
            0 0 0 58px rgba(255,255,255,.06);
        pointer-events: none;
    }

    .hero-tag {
        color: #fff3b0;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.15em;
    }

    .hero h2 {
        color: white !important;
        font-size: clamp(1.7rem, 3vw, 2.5rem);
        line-height: 1.2;
        margin: 12px 0;
    }

    .hero p {
        color: #fff4fc !important;
        max-width: 680px;
        font-size: 0.96rem;
        line-height: 1.65;
        margin-bottom: 0;
    }

    /* SECTION HEADINGS */

    /* SECTION HEADINGS */

    .section-heading {
        margin: 25px 0 10px;
    }

    .section-heading h3 {
        margin: 0;
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 800;
    }

    .section-note {
        color: #b4b8cc;
        font-size: 0.82rem;
        margin-top: 5px;
    }

    /* PANELS */

    .panel {
        background: rgba(21, 24, 43, 0.90);
        border: 1px solid #343751;
        border-radius: 19px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
    }

    /* CHARTS */

    [data-testid="stPlotlyChart"] {
        background: rgba(21, 24, 43, 0.88);
        border: 1px solid #343751;
        border-radius: 16px;
        padding: 5px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    /* DATA TABLES */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #343751;
        border-radius: 13px;
        overflow: hidden;
    }

    /* FORM */

    div[data-testid="stForm"] {
        background: rgba(21, 24, 43, 0.94);
        border: 1px solid #3a3d5c;
        border-radius: 19px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
    }

    /* BUTTONS */

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe, #e0e7ff);
        color: #ffffff !important;
        border: 0;
        border-radius: 11px;
        padding: 0.7rem 1.1rem;
        font-weight: 800;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 0 20px rgba(236, 72, 153, 0.30);
    }

    /* INPUTS */

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: #20243b !important;
        color: #ffffff !important;
        border: 1px solid #414564 !important;
        border-radius: 10px !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background: #20243b !important;
        color: #ffffff !important;
        border-color: #414564 !important;
        border-radius: 10px !important;
    }

    [data-testid="stSelectbox"] *,
    [data-testid="stMultiSelect"] * {
        color: #f8fafc;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background: #20243b !important;
    }

    [role="option"] {
        color: #ffffff !important;
    }

    /* CHECKBOXES AND OTHER WIDGET LABELS */

    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: #e2e5f2 !important;
    }

    /* ALERTS */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stAlert"] p {
        color: inherit;
    }

    /* EXPANDERS */

    [data-testid="stExpander"] {
        background: rgba(21, 24, 43, 0.92);
        border: 1px solid #343751;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #f8fafc !important;
    }

    /* JSON AND CODE AREAS */

    [data-testid="stJson"],
    pre,
    code {
        border-radius: 8px;
    }

    /* FOOTER */

    .footnote {
        color: #aeb3ca;
        font-size: 0.77rem;
        line-height: 1.6;
        padding-top: 20px;
        border-top: 1px solid #343751;
        margin-top: 30px;
    }

    hr {
        border-color: #343751 !important;
    }

    /* MOBILE */

    @media (max-width: 760px) {
        .block-container {
            padding: 1rem 0.8rem 2rem;
        }

        .hero {
            padding: 23px;
            border-radius: 19px;
        }

        div[data-testid="stSegmentedControl"] label {
            padding: 8px 10px !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div,
        [data-testid="stMetricValue"] span {
            font-size: 1.7rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD SAVED MODELS AND DATA
# ============================================================

@st.cache_resource
def load_models():
    prediction_model = None
    clustering_model = None
    metadata = {}

    prediction_path = BASE / "models" / "best_model.joblib"
    clustering_path = BASE / "models" / "kmeans_pipeline.joblib"
    metadata_path = BASE / "results" / "model_metadata.json"

    try:
        if prediction_path.exists():
            prediction_model = joblib.load(prediction_path)

        if clustering_path.exists():
            clustering_model = joblib.load(clustering_path)

        if metadata_path.exists():
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

    except (OSError, ValueError, EOFError, ImportError) as error:
        st.error(f"Unable to load saved model files: {error}")

    return prediction_model, clustering_model, metadata


@st.cache_data
def load_customer_data():
    path = BASE / "data" / "processed" / "test.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Unable to read the customer dataset: {error}")
        return pd.DataFrame()


@st.cache_data
def load_model_comparison():
    path = BASE / "results" / "model_comparison.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Unable to read model comparison results: {error}")
        return pd.DataFrame()


model, kmeans, metadata = load_models()
customer_df = load_customer_data()
comparison_df = load_model_comparison()

RFM_FEATURES = ["recency_days", "frequency", "monetary"]

SEGMENT_COLORS = {
    "High value": "#ff4fa3",
    "Frequent": "#a78bfa",
    "At-risk": "#ffadad",
    "Occasional": "#f6b6dc",
    "Unassigned": "#8b90a8",
}

# ============================================================
# HELPERS
# ============================================================

def section(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-heading">
            <h3>{title}</h3>
            <div class="section-note">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=25, b=12),
        font=dict(
            family="DM Sans, sans-serif",
            color="#e7e9f5",
            size=12,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#e7e9f5"),
        ),
        hoverlabel=dict(
            bgcolor="#29213f",
            bordercolor="#ec4899",
            font_color="#ffffff",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(180,184,204,0.12)",
        zeroline=False,
        color="#d8daea",
        linecolor="#454962",
    )
    fig.update_yaxes(
        gridcolor="rgba(180,184,204,0.12)",
        zeroline=False,
        color="#d8daea",
        linecolor="#454962",
    )
    return fig


def numeric_series(data, column):
    if column not in data.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(data[column], errors="coerce").dropna()


def build_segments(data):
    result = data.copy()

    if result.empty:
        result["Cluster"] = pd.Series(dtype="int64")
        result["Segment"] = pd.Series(dtype="object")
        return result

    if not all(column in result.columns for column in RFM_FEATURES):
        result["Cluster"] = -1
        result["Segment"] = "Unassigned"
        return result

    usable = result[RFM_FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = usable.notna().all(axis=1)
    result["Cluster"] = -1

    if kmeans is not None and valid.any():
        try:
            result.loc[valid, "Cluster"] = kmeans.predict(
                usable.loc[valid, RFM_FEATURES]
            ).astype(int)
        except (ValueError, TypeError, AttributeError):
            pass

    if (result["Cluster"] == -1).all():
        result["Segment"] = "Unassigned"
        return result

    valid_clusters = result[result["Cluster"] >= 0]
    stats = valid_clusters.groupby("Cluster")[RFM_FEATURES].mean()
    names = {}

    if not stats.empty:
        remaining = list(stats.index)

        at_risk = stats["recency_days"].idxmax()
        names[at_risk] = "At-risk"
        remaining.remove(at_risk)

        if remaining:
            high_value = stats.loc[remaining, "monetary"].idxmax()
            names[high_value] = "High value"
            remaining.remove(high_value)

        if remaining:
            frequent = stats.loc[remaining, "frequency"].idxmax()
            names[frequent] = "Frequent"
            remaining.remove(frequent)

        for cluster_id in remaining:
            names[cluster_id] = "Occasional"

    result["Segment"] = result["Cluster"].map(names).fillna("Unassigned")
    return result


segmented_df = build_segments(customer_df)

feature_columns = metadata.get("feature_columns", [])

if not feature_columns and not customer_df.empty:
    excluded = {
        "CustomerID", "repeat_purchase", "Cluster", "Segment"
    }
    feature_columns = [
        column for column in customer_df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(customer_df[column])
    ]

feature_columns = [
    column for column in feature_columns
    if column in customer_df.columns
]

# ============================================================
# BRAND AND HORIZONTAL NAVIGATION
# ============================================================

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">S</div>
        <div>
            <div class="brand-name">ShopSense</div>
            <div class="brand-caption">
                Customer Personality & Shopping Pattern Analyzer
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.segmented_control(
    "Navigation",
    [
        "✨ Overview",
        "👥 Customer Segments",
        "🎯 Purchase Prediction",
        "📊 Model Insights",
        "🔎 Data Explorer",
    ],
    default="✨ Overview",
    selection_mode="single",
    label_visibility="collapsed",
    key="top_navigation",
)

page_names = {
    "✨ Overview": "Overview",
    "👥 Customer Segments": "Customer Segments",
    "🎯 Purchase Prediction": "Purchase Prediction",
    "📊 Model Insights": "Model Insights",
    "🔎 Data Explorer": "Data Explorer",
}

page = page_names.get(page, "Overview")

st.markdown(
    '<div class="eyebrow">CUSTOMER ANALYTICS WORKSPACE</div>',
    unsafe_allow_html=True,
)

if customer_df.empty:
    st.error(
        "Customer data could not be loaded. Check that "
        "data/processed/test.csv exists and contains the processed "
        "customer-level data produced by your pipeline."
    )
    st.stop()

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown(
        '<div class="page-title">Understand your customers.</div>'
        '<div class="page-subtitle">'
        'Explore customer records, RFM patterns, K-means segments, '
        'and repeat-purchase model results.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">CUSTOMER INTELLIGENCE</div>
            <h2>From shopping patterns<br>to customer insights.</h2>
            <p>
                ShopSense uses recency, frequency and monetary value
                to explore customer behavior and support repeat-purchase
                predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Customer records", f"{len(customer_df):,}")

    if "recency_days" in customer_df.columns:
        values = numeric_series(customer_df, "recency_days")
        k2.metric(
            "Average recency",
            f"{values.mean():.1f} days" if not values.empty else "N/A",
        )
    else:
        k2.metric("Average recency", "N/A")

    if "frequency" in customer_df.columns:
        values = numeric_series(customer_df, "frequency")
        k3.metric(
            "Average frequency",
            f"{values.mean():.1f}" if not values.empty else "N/A",
        )
    else:
        k3.metric("Average frequency", "N/A")

    if "monetary" in customer_df.columns:
        values = numeric_series(customer_df, "monetary")
        k4.metric(
            "Average monetary value",
            f"{values.mean():,.2f}" if not values.empty else "N/A",
        )
    else:
        k4.metric("Average monetary value", "N/A")

    st.caption(
        "Metrics describe the loaded customer dataset only. "
        "Monetary values use the units in the processed data."
    )

    left, right = st.columns([1, 1.2])

    with left:
        section(
            "Customer segments",
            "Distribution from available K-means assignments",
        )

        assigned = segmented_df[
            segmented_df["Segment"] != "Unassigned"
        ]

        if not assigned.empty:
            counts = (
                assigned["Segment"]
                .value_counts()
                .rename_axis("Segment")
                .reset_index(name="Customers")
            )

            fig = px.pie(
                counts,
                names="Segment",
                values="Customers",
                hole=0.64,
                color="Segment",
                color_discrete_map=SEGMENT_COLORS,
            )
            fig.update_traces(
                hovertemplate=(
                    "%{label}<br>Customers: %{value:,}"
                    "<br>Share: %{percent}<extra></extra>"
                ),
                textfont=dict(color="#ffffff"),
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e7e9f5",
            )
            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info(
                "No valid K-means segment assignments are available. "
                "Check the saved clustering model and RFM columns."
            )

    with right:
        section(
            "Frequency vs. monetary value",
            "Each point represents one customer record",
        )

        required = {"frequency", "monetary", "Segment"}

        if required.issubset(segmented_df.columns):
            plot_df = segmented_df[
                segmented_df["Segment"] != "Unassigned"
            ].copy()

            if not plot_df.empty:
                hover_columns = [
                    column for column in [
                        "CustomerID", "recency_days",
                        "frequency", "monetary",
                    ]
                    if column in plot_df.columns
                ]

                fig = px.scatter(
                    plot_df,
                    x="frequency",
                    y="monetary",
                    color="Segment",
                    hover_data=hover_columns,
                    color_discrete_map=SEGMENT_COLORS,
                    labels={
                        "frequency": "Purchase frequency",
                        "monetary": "Monetary value",
                        "Segment": "Customer segment",
                    },
                )
                fig.update_traces(
                    marker=dict(
                        opacity=0.85,
                        size=8,
                        line=dict(width=0.5, color="#ffffff"),
                    )
                )
                st.plotly_chart(
                    style_chart(fig, 350),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            else:
                st.info("No assigned customer segments to plot.")
        else:
            st.info("Frequency and monetary columns are required.")

    section(
        "Customer-level RFM summary",
        "Average recency, frequency and monetary value by segment",
    )

    if all(column in segmented_df.columns for column in RFM_FEATURES):
        summary = (
            segmented_df.groupby("Segment", as_index=False)
            .agg(
                Customers=("Segment", "size"),
                Average_Recency=("recency_days", "mean"),
                Average_Frequency=("frequency", "mean"),
                Average_Monetary=("monetary", "mean"),
            )
            .round(2)
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("RFM columns are not all available in the dataset.")

# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

elif page == "Customer Segments":
    st.markdown(
        '<div class="page-title">Customer segmentation</div>'
        '<div class="page-subtitle">'
        'Explore the customer groups identified by the saved '
        'K-means pipeline using available RFM features.'
        '</div>',
        unsafe_allow_html=True,
    )

    counts = segmented_df["Segment"].value_counts()
    columns = st.columns(4)

    for column, name in zip(
        columns,
        ["High value", "Frequent", "At-risk", "Occasional"],
    ):
        count = int(counts.get(name, 0))
        share = count / max(len(segmented_df), 1)

        with column:
            st.metric(name, f"{count:,}", f"{share:.1%} of records")

    left, right = st.columns([1.2, 1])

    with left:
        section(
            "Segment map",
            "Purchase frequency against monetary value",
        )

        if all(
            column in segmented_df.columns
            for column in RFM_FEATURES
        ):
            plot_df = segmented_df[
                segmented_df["Segment"] != "Unassigned"
            ]

            if not plot_df.empty:
                hover_columns = [
                    column for column in ["CustomerID", "recency_days"]
                    if column in plot_df.columns
                ]

                fig = px.scatter(
                    plot_df,
                    x="frequency",
                    y="monetary",
                    color="Segment",
                    hover_data=hover_columns,
                    color_discrete_map=SEGMENT_COLORS,
                    labels={
                        "frequency": "Purchase frequency",
                        "monetary": "Monetary value",
                    },
                )
                fig.update_traces(
                    marker=dict(
                        opacity=0.85,
                        size=8,
                        line=dict(width=0.5, color="#ffffff"),
                    )
                )
                st.plotly_chart(
                    style_chart(fig, 390),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            else:
                st.info("No valid segment data to plot.")
        else:
            st.info("Required RFM columns are missing.")

    with right:
        section("Segment sizes", "Number of customer records")

        segment_counts = (
            segmented_df["Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
            .sort_values("Customers")
        )

        fig = px.bar(
            segment_counts,
            x="Customers",
            y="Segment",
            orientation="h",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            text="Customers",
        )
        fig.update_traces(
            textposition="outside",
            textfont=dict(color="#f8fafc"),
            cliponaxis=False,
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title=None,
            margin=dict(l=12, r=40, t=25, b=12),
        )

        st.plotly_chart(
            style_chart(fig, 390),
            use_container_width=True,
            config={"displaylogo": False},
        )

    section("Segment profile", "Mean customer-level RFM values")

    if all(
        column in segmented_df.columns
        for column in RFM_FEATURES
    ):
        summary = (
            segmented_df.groupby("Segment", as_index=False)
            .agg(
                Customers=("Segment", "size"),
                Average_Recency=("recency_days", "mean"),
                Average_Frequency=("frequency", "mean"),
                Average_Monetary=("monetary", "mean"),
            )
            .round(2)
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    section("Explore customer records")

    selected_segment = st.selectbox(
        "Select a segment",
        ["All segments"]
        + sorted(segmented_df["Segment"].dropna().unique().tolist()),
    )

    filtered = segmented_df.copy()

    if selected_segment != "All segments":
        filtered = filtered[filtered["Segment"] == selected_segment]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(
        f"Showing {len(filtered):,} of {len(segmented_df):,} records."
    )

# ============================================================
# PURCHASE PREDICTION
# ============================================================

elif page == "Purchase Prediction":
    st.markdown(
        '<div class="page-title">Repeat-purchase prediction</div>'
        '<div class="page-subtitle">'
        'Enter a customer profile and use the saved trained model '
        'to estimate the repeat-purchase outcome.'
        '</div>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "The trained model was not found at "
            "models/best_model.joblib. Train the model first."
        )
        st.stop()

    if not feature_columns:
        st.error(
            "No prediction features are available. Check "
            "results/model_metadata.json and the processed dataset."
        )
        st.stop()

    with st.form("prediction_form"):
        st.markdown("#### Customer profile")
        values = {}
        input_columns = st.columns(2)

        for index, feature in enumerate(feature_columns):
            numeric = numeric_series(customer_df, feature)

            if numeric.empty:
                minimum, maximum, default = 0.0, 100.0, 0.0
            else:
                minimum = float(numeric.min())
                maximum = float(numeric.max())
                default = float(numeric.median())

            if minimum == maximum:
                minimum = min(0.0, minimum)
                maximum = maximum + 1.0

            step = 1.0 if maximum - minimum > 20 else 0.1

            with input_columns[index % 2]:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(np.clip(default, minimum, maximum)),
                    step=float(step),
                    key=f"prediction_{feature}",
                )

        submitted = st.form_submit_button(
            "Run prediction",
            use_container_width=True,
        )

    if submitted:
        sample = pd.DataFrame([values], columns=feature_columns)

        try:
            raw_prediction = model.predict(sample)[0]
            prediction = int(raw_prediction)
            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    probability = float(
                        probabilities[classes.index(1)]
                    )

            st.markdown("#### Prediction result")

            left, right = st.columns([1, 1.3])

            with left:
                if probability is not None:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            number={
                                "suffix": "%",
                                "font": {"color": "#ffffff"},
                            },
                            title={
                                "text": "Predicted repeat-purchase probability",
                                "font": {"color": "#e7e9f5"},
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "tickcolor": "#e7e9f5",
                                },
                                "bar": {"color": "#ff4fa3"},
                                "bgcolor": "#20243b",
                                "bordercolor": "#414564",
                                "steps": [
                                    {
                                        "range": [0, 40],
                                        "color": "#392039",
                                    },
                                    {
                                        "range": [40, 70],
                                        "color": "#33294d",
                                    },
                                    {
                                        "range": [70, 100],
                                        "color": "#263b55",
                                    },
                                ],
                            },
                        )
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(l=15, r=15, t=55, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#ffffff",
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
                else:
                    st.metric(
                        "Predicted outcome",
                        "Repeat purchase"
                        if prediction == 1
                        else "No repeat purchase",
                    )

            with right:
                if prediction == 1:
                    st.success("Prediction: Repeat purchase")
                else:
                    st.warning("Prediction: No repeat purchase")

                if probability is not None:
                    st.metric(
                        "Repeat-purchase probability",
                        f"{probability:.1%}",
                    )

                st.caption(
                    "This is a model estimate, not a guarantee of "
                    "future customer behavior."
                )
                st.dataframe(
                    sample,
                    use_container_width=True,
                    hide_index=True,
                )

        except (
            ValueError, TypeError, AttributeError, KeyError
        ) as error:
            st.error(f"Prediction failed: {error}")

# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":
    st.markdown(
        '<div class="page-title">Model performance</div>'
        '<div class="page-subtitle">'
        'Review the saved model-comparison results produced by '
        'the ShopSense training and evaluation pipeline.'
        '</div>',
        unsafe_allow_html=True,
    )

    if comparison_df.empty:
        st.info(
            "Model comparison results are unavailable. Run the "
            "training and evaluation scripts to generate "
            "results/model_comparison.csv."
        )
    else:
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        numeric_columns = comparison_df.select_dtypes(
            include="number"
        ).columns.tolist()

        model_column = next(
            (
                column for column in comparison_df.columns
                if column.lower() in {
                    "model", "model_name", "classifier",
                    "estimator", "algorithm",
                }
            ),
            None,
        )

        f1_column = next(
            (
                column for column in numeric_columns
                if "f1" in column.lower()
            ),
            None,
        )

        if model_column and f1_column:
            section(
                "F1-score comparison",
                "Values taken from saved evaluation results",
            )

            fig = px.bar(
                comparison_df,
                x=model_column,
                y=f1_column,
                color=model_column,
                text_auto=".3f",
                labels={
                    model_column: "Model",
                    f1_column: f1_column.replace("_", " ").title(),
                },
                color_discrete_sequence=[
                    "#ff4fa3", "#a78bfa", "#ffadad"
                ],
            )
            fig.update_traces(
                textposition="outside",
                textfont=dict(color="#f8fafc"),
                cliponaxis=False,
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title=None,
                margin=dict(l=12, r=25, t=25, b=12),
            )

            st.plotly_chart(
                style_chart(fig, 360),
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info(
                "A model-name column and an F1-score column are "
                "needed to draw the comparison chart. The results "
                "table is still available."
            )

    section("Saved model files")

    metadata_path = BASE / "results" / "model_metadata.json"

    a, b, c = st.columns(3)

    a.metric(
        "Prediction model",
        "Loaded" if model is not None else "Not available",
    )
    b.metric(
        "K-means model",
        "Loaded" if kmeans is not None else "Not available",
    )

    if metadata_path.exists():
        try:
            saved_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            feature_count = len(
                saved_metadata.get("feature_columns", [])
            )
        except (OSError, ValueError):
            saved_metadata = {}
            feature_count = 0
    else:
        saved_metadata = {}
        feature_count = 0

    c.metric("Saved input features", feature_count)

    if saved_metadata:
        with st.expander("View saved model metadata"):
            st.json(saved_metadata)

# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    st.markdown(
        '<div class="page-title">Explore customer data</div>'
        '<div class="page-subtitle">'
        'Filter and inspect the customer records available to '
        'the ShopSense dashboard.'
        '</div>',
        unsafe_allow_html=True,
    )

    filters = st.columns([1, 1, 1.4])

    with filters[0]:
        segment_options = ["All"] + sorted(
            segmented_df["Segment"].dropna().unique().tolist()
        )
        segment_filter = st.selectbox(
            "Customer segment",
            segment_options,
        )

    with filters[1]:
        if "repeat_purchase" in segmented_df.columns:
            repeat_filter = st.selectbox(
                "Repeat purchase",
                ["All", "Yes", "No"],
            )
        else:
            repeat_filter = "All"
            st.caption(
                "Repeat-purchase labels are not present in this file."
            )

    with filters[2]:
        search_id = st.text_input(
            "Search customer ID",
            placeholder="Enter an ID",
        )

    filtered = segmented_df.copy()

    if segment_filter != "All":
        filtered = filtered[filtered["Segment"] == segment_filter]

    if repeat_filter != "All" and "repeat_purchase" in filtered.columns:
        target = 1 if repeat_filter == "Yes" else 0
        filtered = filtered[
            pd.to_numeric(
                filtered["repeat_purchase"], errors="coerce"
            ) == target
        ]

    if search_id.strip() and "CustomerID" in filtered.columns:
        filtered = filtered[
            filtered["CustomerID"]
            .astype(str)
            .str.contains(search_id.strip(), case=False, na=False)
        ]

    st.markdown(
        f"**{len(filtered):,} records match your filters.**"
    )
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footnote">
        ShopSense · Customer Personality & Shopping Pattern Analyzer
        <br>
        Visualizations use the available processed customer data,
        saved cluster assignments, or model evaluation results.
        Missing business metrics are not replaced with synthetic values.
    </div>
    """,
    unsafe_allow_html=True,
)
