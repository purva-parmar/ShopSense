
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent

# -----------------------------------------------
# Page configuration
# -----------------------------------------------

st.set_page_config(
    page_title="ShopSense | Customer Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------
# Load CSS
# -----------------------------------------------

css_path = BASE / "style.css"

if css_path.exists():
    with open(css_path, encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )

# -----------------------------------------------
# Load models and data
# -----------------------------------------------

@st.cache_resource
def load_models():
    prediction_model = joblib.load(
        BASE / "models" / "best_model.joblib"
    )
    clustering_model = joblib.load(
        BASE / "models" / "kmeans_pipeline.joblib"
    )

    with open(
        BASE / "results" / "model_metadata.json",
        encoding="utf-8"
    ) as file:
        model_metadata = json.load(file)

    return prediction_model, clustering_model, model_metadata


@st.cache_data
def load_test_data():
    return pd.read_csv(
        BASE / "data" / "processed" / "test.csv"
    )


@st.cache_data
def load_comparison():
    path = BASE / "results" / "model_comparison.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


model, kmeans, metadata = load_models()
test_df = load_test_data()
comparison = load_comparison()

feature_columns = metadata["feature_columns"]

cluster_features = [
    "recency_days",
    "frequency",
    "monetary"
]

# -----------------------------------------------
# Prepare customer segments
# -----------------------------------------------

@st.cache_data
def create_segment_data(data):
    result = data.copy()

    features = result[cluster_features].copy()
    valid_rows = features.notna().all(axis=1)

    result["Cluster"] = pd.NA

    if valid_rows.any():
        result.loc[valid_rows, "Cluster"] = (
            kmeans.predict(features.loc[valid_rows]).astype(int)
        )

    result["Cluster"] = result["Cluster"].astype("Int64")

    valid = result.dropna(subset=["Cluster"]).copy()
    valid["Cluster"] = valid["Cluster"].astype(int)

    if valid.empty:
        return valid, {}, pd.DataFrame()

    stats = valid.groupby("Cluster")[cluster_features].mean()

    remaining = list(stats.index)
    names = {}

    # Heuristic descriptions based on cluster averages
    at_risk = stats.loc[remaining, "recency_days"].idxmax()
    names[at_risk] = "At-Risk Customers"
    remaining.remove(at_risk)

    if remaining:
        high_value = stats.loc[remaining, "monetary"].idxmax()
        names[high_value] = "High-Spending Customers"
        remaining.remove(high_value)

    if remaining:
        frequent = stats.loc[remaining, "frequency"].idxmax()
        names[frequent] = "Frequent Buyers"
        remaining.remove(frequent)

    for cluster_id in remaining:
        names[cluster_id] = "Occasional Buyers"

    valid["Segment"] = valid["Cluster"].map(names)

    summary = (
        valid.groupby(["Cluster", "Segment"])
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency_Days=("recency_days", "mean"),
            Avg_Frequency=("frequency", "mean"),
            Avg_Monetary=("monetary", "mean")
        )
        .reset_index()
    )

    summary = summary.round(2)

    return valid, names, summary


cluster_df, cluster_names, cluster_summary = create_segment_data(
    test_df
)

# Fixed colors for consistent segment identity
SEGMENT_COLORS = {
    "Frequent Buyers": "#16877D",
    "At-Risk Customers": "#E87965",
    "High-Spending Customers": "#8B73C7",
    "Occasional Buyers": "#D29A2E"
}

SEGMENT_BACKGROUNDS = {
    "Frequent Buyers": "#E2F4EF",
    "At-Risk Customers": "#FFF0EB",
    "High-Spending Customers": "#F0EBFF",
    "Occasional Buyers": "#FFF4D9"
}

# -----------------------------------------------
# Top navigation and brand
# -----------------------------------------------

st.markdown(
    """
    <div class="brand-header">
        <div class="brand-icon">🛍️</div>
        <div>
            <div class="brand-name">ShopSense</div>
            <div class="brand-subtitle">
                CUSTOMER PERSONALITY & SHOPPING ANALYTICS
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.radio(
    "Main navigation",
    [
        "Overview",
        "Customer Segmentation",
        "Purchase Prediction",
        "Model Performance"
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation"
)

st.divider()

# -----------------------------------------------
# Overview
# -----------------------------------------------

if page == "Overview":

    st.markdown(
        """
        <div class="eyebrow">YOUR RETAIL INTELLIGENCE SPACE</div>
        <h1 class="page-heading">Understand your customers.</h1>
        <p class="page-description">
            Explore shopping patterns, discover customer segments,
            and predict repeat-purchase behavior with machine learning.
        </p>
        """,
        unsafe_allow_html=True
    )

    total_customers = len(test_df)
    total_features = len(feature_columns)
    total_segments = cluster_df["Cluster"].nunique()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Customers Analyzed", f"{total_customers:,}")

    with col2:
        st.metric("Prediction Features", total_features)

    with col3:
        st.metric("Customer Segments", total_segments)

    st.markdown("### Customer Snapshot")
    st.caption(
        "A quick look at repeat-purchase behavior in the test dataset."
    )

    if "repeat_purchase" in test_df.columns:
        counts = (
            test_df["repeat_purchase"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        col1, col2 = st.columns([1.15, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")

            bars = ax.bar(
                ["No Repeat Purchase", "Repeat Purchase"],
                [counts.loc[0], counts.loc[1]],
                color=["#E87965", "#16877D"],
                width=0.55
            )

            ax.set_ylabel("Number of Customers", color="#555B66")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#E8E3DA")
            ax.tick_params(axis="x", rotation=0, colors="#454B55")
            ax.tick_params(axis="y", colors="#747985")
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)

            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{int(bar.get_height())}",
                    ha="center",
                    va="bottom",
                    color="#252A35",
                    fontweight="bold"
                )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            repeat_count = int(counts.loc[1])
            no_repeat_count = int(counts.loc[0])

            st.markdown(
                f"""
                <div class="insight-card teal-card">
                    <div class="insight-label">REPEAT PURCHASE</div>
                    <div class="insight-number">{repeat_count:,}</div>
                    <div class="insight-description">
                        Customers labeled as repeat purchasers
                        in the test data.
                    </div>
                </div>
                <div class="insight-card coral-card">
                    <div class="insight-label">NO REPEAT PURCHASE</div>
                    <div class="insight-number">{no_repeat_count:,}</div>
                    <div class="insight-description">
                        Customers labeled as non-repeat purchasers
                        in the test data.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Explore Customer Segments")

    segment_cols = st.columns(4)

    for index, segment in enumerate(SEGMENT_COLORS):
        count = int(
            (cluster_df["Segment"] == segment).sum()
        ) if not cluster_df.empty else 0

        with segment_cols[index]:
            st.markdown(
                f"""
                <div class="segment-card"
                     style="background:{SEGMENT_BACKGROUNDS[segment]};
                            border-top:4px solid {SEGMENT_COLORS[segment]};">
                    <div class="segment-count"
                         style="color:{SEGMENT_COLORS[segment]}">
                        {count}
                    </div>
                    <div class="segment-name">{segment}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.caption(
        "Customer segment names are descriptive interpretations "
        "of cluster averages, not ground-truth labels."
    )

    with st.expander("View sample customer records"):
        st.dataframe(
            test_df.head(10),
            use_container_width=True,
            hide_index=True
        )

# -----------------------------------------------
# Customer Segmentation
# -----------------------------------------------

elif page == "Customer Segmentation":

    st.markdown(
        """
        <div class="eyebrow">CUSTOMER INTELLIGENCE</div>
        <h1 class="page-heading">Know every segment.</h1>
        <p class="page-description">
            Explore groups formed using recency, purchase frequency,
            and monetary value.
        </p>
        """,
        unsafe_allow_html=True
    )

    if cluster_df.empty:
        st.warning("No valid customers are available for clustering.")
        st.stop()

    # Four colored segment cards
    segment_cols = st.columns(4)

    for index, segment in enumerate(SEGMENT_COLORS):
        count = int((cluster_df["Segment"] == segment).sum())

        with segment_cols[index]:
            st.markdown(
                f"""
                <div class="segment-card"
                     style="background:{SEGMENT_BACKGROUNDS[segment]};
                            border-top:4px solid {SEGMENT_COLORS[segment]};">
                    <div class="segment-count"
                         style="color:{SEGMENT_COLORS[segment]}">
                        {count}
                    </div>
                    <div class="segment-name">{segment}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Segment Distribution")

    counts = (
        cluster_df["Segment"]
        .value_counts()
        .reindex(list(SEGMENT_COLORS.keys()), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[SEGMENT_COLORS[name] for name in counts.index],
        width=0.62
    )

    ax.set_ylabel("Number of Customers", color="#555B66")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#E8E3DA")
    ax.tick_params(axis="x", rotation=12, colors="#454B55")
    ax.tick_params(axis="y", colors="#747985")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            color="#252A35",
            fontweight="bold"
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Segment Characteristics")

    display_summary = cluster_summary.drop(
        columns=["Cluster"],
        errors="ignore"
    )

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "At-Risk Customers are identified heuristically by higher average "
        "recency. High-Spending Customers are identified by higher average "
        "monetary value among the remaining clusters. These labels are "
        "interpretations of K-means clusters, not verified customer categories."
    )

    st.markdown("### Explore Individual Customers")

    selected_segment = st.selectbox(
        "Filter by segment",
        ["All Segments"] + list(SEGMENT_COLORS.keys())
    )

    if selected_segment == "All Segments":
        filtered = cluster_df
    else:
        filtered = cluster_df[
            cluster_df["Segment"] == selected_segment
        ]

    columns_to_show = [
        "CustomerID",
        "recency_days",
        "frequency",
        "monetary",
        "Cluster",
        "Segment"
    ]

    st.dataframe(
        filtered[columns_to_show],
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"Showing {len(filtered):,} customers.")

# -----------------------------------------------
# Purchase Prediction
# -----------------------------------------------

elif page == "Purchase Prediction":

    st.markdown(
        """
        <div class="eyebrow">PREDICTIVE ANALYTICS</div>
        <h1 class="page-heading">Predict the next purchase.</h1>
        <p class="page-description">
            Enter customer shopping statistics to estimate repeat-purchase
            behavior using the trained machine-learning model.
        </p>
        """,
        unsafe_allow_html=True
    )

    missing_features = [
        feature for feature in feature_columns
        if feature not in test_df.columns
    ]

    if missing_features:
        st.error(f"Missing prediction features: {missing_features}")
        st.stop()

    st.info(
        "Input fields start with median values from the test dataset. "
        "Change them to explore different customer profiles."
    )

    values = {}

    with st.form("prediction_form"):

        st.markdown("### Customer Shopping Profile")
        input_cols = st.columns(2)

        for index, feature in enumerate(feature_columns):

            feature_values = pd.to_numeric(
                test_df[feature],
                errors="coerce"
            ).dropna()

            if feature_values.empty:
                minimum, maximum, default = 0.0, 100.0, 0.0
            else:
                minimum = float(feature_values.min())
                maximum = float(feature_values.max())
                default = float(feature_values.median())

            if minimum == maximum:
                minimum = min(0.0, minimum)
                maximum = maximum + 1.0

            with input_cols[index % 2]:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default, minimum), maximum),
                    step=1.0,
                    format="%.2f",
                    key=f"input_{feature}"
                )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:

        sample = pd.DataFrame(
            [values],
            columns=feature_columns
        )

        try:
            prediction = int(model.predict(sample)[0])
            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    probability = float(
                        probabilities[classes.index(1)]
                    )

            st.markdown("### Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                if probability is not None:
                    st.metric(
                        "Repeat-Purchase Probability",
                        f"{probability:.2%}"
                    )
                else:
                    st.metric(
                        "Repeat-Purchase Probability",
                        "Unavailable"
                    )

            with col2:
                st.metric(
                    "Predicted Outcome",
                    "Repeat Purchase" if prediction == 1
                    else "No Repeat Purchase"
                )

            if prediction == 1:
                st.success(
                    "The model predicts a repeat purchase for this profile."
                )
            else:
                st.warning(
                    "The model predicts no repeat purchase for this profile."
                )

            st.caption(
                "This is a model estimate, not a guarantee of future behavior."
            )

            with st.expander("Review submitted feature values"):
                st.dataframe(
                    sample,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# -----------------------------------------------
# Model Performance
# -----------------------------------------------

elif page == "Model Performance":

    st.markdown(
        """
        <div class="eyebrow">MODEL EVALUATION</div>
        <h1 class="page-heading">Measure model performance.</h1>
        <p class="page-description">
            Compare trained models and review available evaluation results.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Model Comparison")

    if not comparison.empty:
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No model comparison data was found.")

    result_dir = BASE / "results"

    st.markdown("### Confusion Matrices")

    confusion_images = sorted(
        result_dir.glob("confusion_matrix_*.png")
    )

    if confusion_images:
        for start in range(0, len(confusion_images), 2):
            image_cols = st.columns(2)

            for offset, image_path in enumerate(
                confusion_images[start:start + 2]
            ):
                with image_cols[offset]:
                    st.image(
                        str(image_path),
                        caption=image_path.stem.replace(
                            "_", " "
                        ).title(),
                        use_container_width=True
                    )
    else:
        st.info("No confusion matrix images were found.")

    st.markdown("### F1-Score Comparisons")

    f1_images = [
        result_dir / "validation_f1_comparison.png",
        result_dir / "test_f1_comparison.png"
    ]

    available_images = [
        path for path in f1_images if path.exists()
    ]

    if available_images:
        image_cols = st.columns(len(available_images))

        for index, image_path in enumerate(available_images):
            with image_cols[index]:
                st.image(
                    str(image_path),
                    caption=image_path.stem.replace(
                        "_", " "
                    ).title(),
                    use_container_width=True
                )
    else:
        st.info("No F1-score comparison charts were found.")

    st.caption(
        "Interpret model performance using the reported evaluation metrics "
        "and the corresponding test results."
    )

# -----------------------------------------------
# Footer
# -----------------------------------------------

st.markdown(
    """
    <div class="app-footer">
        <span>ShopSense</span>
        <span>Customer Intelligence · Machine Learning</span>
    </div>
    """,
    unsafe_allow_html=True
)
