import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import joblib
import os
import sys
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AeroPredict AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# 2. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "aeropredict_lstm.keras"
CONFIG_PATH = BASE_DIR / "models" / "aeropredict_config.pkl"

REPORTS_DIR = BASE_DIR / "reports"

COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
SHAP_PATH = REPORTS_DIR / "shap_feature_importance.csv"

# ==================================================
# 3. LOAD MODEL AND CONFIGURATION
# ==================================================


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_config():
    return joblib.load(CONFIG_PATH)


try:
    lstm_model = load_model()
    config = load_config()

    scaler = config["scaler"]
    feature_columns = config["feature_columns"]
    sequence_length = config["sequence_length"]

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# ==================================================
# 4. DATASET COLUMNS
# ==================================================

columns = ["Engine_ID", "Cycle", "Setting_1", "Setting_2", "Setting_3"]

for i in range(1, 22):
    columns.append(f"Sensor_{i}")


# ==================================================
# 5. CUSTOM CSS
# ==================================================


st.markdown(
    """
<style>

/* Main application */

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            #172b4d 0%,
            #0b1120 45%,
            #080e1b 100%
        );

    color: #e2e8f0;
}

/* Main content spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #101c31,
        #0b1424
    );

    border-right: 1px solid #263750;
}

[data-testid="stSidebar"] h2 {
    color: #60a5fa;
    font-weight: 800;
}

/* Main heading */

.main-title {
    font-size: 42px;
    font-weight: 850;
    letter-spacing: -1.5px;
    color: #f1f5f9;
}

.subtitle {
    font-size: 15px;
    color: #94a3b8;
    margin-top: -8px;
}

/* Section headings */

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(30, 48, 75, 0.95),
        rgba(16, 30, 51, 0.95)
    );

    border: 1px solid rgba(96, 165, 250, 0.18);

    padding: 20px;

    border-radius: 14px;

    box-shadow: 0 8px 25px rgba(0,0,0,0.18);

    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #38bdf8;
    transform: translateY(-3px);
}

/* Metric labels */

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-size: 14px;
}

/* Metric values */

div[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-weight: 750;
}

/* Buttons */

.stButton button,
.stDownloadButton button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #0891b2
    );

    color: white;

    border: none;

    border-radius: 9px;

    padding: 10px 20px;

    font-weight: 650;

    transition: all 0.2s ease;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #0e7490
    );

    border: none;
    transform: translateY(-2px);
}

/* Select boxes */

div[data-baseweb="select"] > div {
    background-color: #172338;
    border-color: #334155;
    border-radius: 9px;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: rgba(23, 35, 56, 0.65);
    border: 1px dashed #3b82f6;
    border-radius: 12px;
    padding: 15px;
}

/* Dataframes */

[data-testid="stDataFrame"] {
    border: 1px solid #263750;
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dividers */

hr {
    border-color: rgba(148, 163, 184, 0.15);
}

/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 20px;
}

/* Smooth appearance */

section[data-testid="stMain"] {
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ==================================================
# 6. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown("## ✈️ AeroPredict AI")

st.sidebar.caption("Aircraft Engine Health Intelligence")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Engine Monitoring",
        "Model Comparison",
        "Explainable AI",
        "Model Information",
        "Manual RUL Prediction",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Model Status")
st.sidebar.success("LSTM Model Loaded")

st.sidebar.caption("NASA C-MAPSS | FD001")

st.sidebar.markdown("### Model Architecture")
st.sidebar.info("Long Short-Term Memory (LSTM)")

st.sidebar.markdown(f"Sequence Length: {sequence_length} cycles")


# ==================================================
# 7. MAIN HEADER
# ==================================================

st.markdown(
    """
<div class="main-title">
    ✈ AeroPredict AI
</div>

<div class="subtitle">
    INTELLIGENT AIRCRAFT ENGINE HEALTH MONITORING
    &nbsp; | &nbsp; PREDICTIVE MAINTENANCE
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# ==================================================
# 8. MANUAL RUL PREDICTION PAGE
# ==================================================

# This page uses the separately saved LSTM and tuned XGBoost models
# from notebooks/models. It is placed before dataset upload so it can
# run without uploading test_FD001.txt.

MANUAL_MODEL_DIR = BASE_DIR / "notebooks" / "models"

TRAIN_DATA_PATH = BASE_DIR / "dataset" / "raw" / "train_FD001.txt"


@st.cache_data
def load_training_data():
    dataset_columns = ["Engine_ID", "Cycle", "Setting_1", "Setting_2", "Setting_3"] + [
        f"Sensor_{i}" for i in range(1, 22)
    ]

    df = pd.read_csv(TRAIN_DATA_PATH, sep=r"\s+", header=None)
    df.columns = dataset_columns

    return df.sort_values(["Engine_ID", "Cycle"])


@st.cache_resource
def load_manual_assets(model_dir):
    manual_lstm = tf.keras.models.load_model(model_dir / "lstm_model.keras")
    manual_xgb = joblib.load(model_dir / "xgboost_model.pkl")
    manual_lstm_scaler = joblib.load(model_dir / "lstm_scaler.pkl")
    manual_features = joblib.load(model_dir / "feature_columns.pkl")
    return manual_lstm, manual_xgb, manual_lstm_scaler, manual_features


if page == "Manual RUL Prediction":

    st.markdown("## Manual RUL Prediction")
    st.markdown(
        "Enter engine operating measurements to estimate Remaining Useful Life. "
        "This page uses the models saved from the FD001 training workflow."
    )

    st.info(
        "Use realistic sensor readings in the same units and feature order as "
        "the NASA C-MAPSS FD001 training data. These predictions are for "
        "project demonstration and are not certified maintenance advice."
    )

    try:
        manual_lstm, manual_xgb, manual_lstm_scaler, manual_features = (
            load_manual_assets(MANUAL_MODEL_DIR)
        )
    except Exception as e:
        st.error(
            "Could not load the manual prediction models. Check that these files "
            f"exist in: {MANUAL_MODEL_DIR}\n\nDetails: {e}"
        )
        st.stop()

    model_choice = st.selectbox(
        "Choose prediction model",
        ["LSTM", "Tuned XGBoost"],
        help=(
            "LSTM requires 30 consecutive cycles. Tuned XGBoost uses one "
            "feature row representing the latest cycle."
        ),
    )

    st.divider()

    if model_choice == "LSTM":

        st.markdown("### Enter 30 consecutive operating cycles")
        st.caption(
            "Fill all 30 rows. Each row represents one consecutive cycle, "
            "and the columns must contain the 18 selected model features. "
            "The final row should be the latest cycle."
        )

        # Load real sample measurements from the FD001 training dataset.
        try:
            training_data = load_training_data()

            sample_engine_id = st.selectbox(
                "Choose a sample engine",
                sorted(training_data["Engine_ID"].unique()),
                key="lstm_sample_engine",
            )

            if st.button("Load Sample Data for LSTM"):
                sample_rows = (
                    training_data.loc[
                        training_data["Engine_ID"] == sample_engine_id,
                        manual_features,
                    ]
                    .tail(30)
                    .reset_index(drop=True)
                )

                if len(sample_rows) != 30:
                    st.warning("The selected engine does not have 30 cycles.")
                else:
                    # Clear the editor's previous widget state so the sample
                    # values are displayed in the editor after rerunning.
                    st.session_state["lstm_sample_values"] = sample_rows
                    if "manual_lstm_input" in st.session_state:
                        del st.session_state["manual_lstm_input"]
                    st.rerun()

        except Exception as e:
            st.error(f"Could not load the FD001 training dataset: {e}")

        blank_input = pd.DataFrame(
            np.nan,
            index=range(30),
            columns=manual_features,
        )

        editor_data = st.session_state.get("lstm_sample_values", blank_input)

        manual_input = st.data_editor(
            editor_data,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            key="manual_lstm_input",
            column_config={
                feature: st.column_config.NumberColumn(
                    feature,
                    help=f"Enter the {feature} measurement for this cycle.",
                    format="%.6f",
                )
                for feature in manual_features
            },
        )

        if st.button("Predict RUL with LSTM", type="primary"):

            if manual_input.isnull().values.any():
                st.error("Please fill in every cell in all 30 rows before predicting.")
            else:
                try:
                    values = manual_input[manual_features].astype(float).values
                    scaled_values = manual_lstm_scaler.transform(values)
                    sequence = scaled_values.reshape(1, 30, len(manual_features))
                    prediction = float(manual_lstm.predict(sequence, verbose=0)[0][0])
                    prediction = max(0.0, prediction)

                    st.divider()
                    result_col1, result_col2 = st.columns(2)
                    result_col1.metric(
                        "Predicted Remaining Useful Life", f"{prediction:.2f} cycles"
                    )

                    if prediction > 60:
                        status = "Healthy"
                        result_col2.success(status)
                    elif prediction > 30:
                        status = "Warning"
                        result_col2.warning(status)
                    else:
                        status = "Critical"
                        result_col2.error(status)

                    st.caption(
                        "Health categories use illustrative project thresholds: "
                        "above 60 = Healthy, above 30 = Warning, and 30 or below = Critical."
                    )
                except Exception as e:
                    st.error(f"LSTM prediction failed: {e}")

    else:

        st.markdown("### Enter latest-cycle feature values")
        st.caption(
            "Enter one latest-cycle measurement for each feature. "
            "Tuned XGBoost was trained on unscaled feature values, so no scaler is applied here."
        )

        # Load the latest real measurement from a selected FD001 engine.
        try:
            training_data = load_training_data()

            sample_engine_id = st.selectbox(
                "Choose a sample engine",
                sorted(training_data["Engine_ID"].unique()),
                key="xgb_sample_engine",
            )

            if st.button("Load Sample Data for XGBoost"):
                sample_row = (
                    training_data.loc[training_data["Engine_ID"] == sample_engine_id]
                    .sort_values("Cycle")
                    .tail(1)
                )

                if sample_row.empty:
                    st.warning("No sample measurements were found for this engine.")
                else:
                    for feature in manual_features:
                        st.session_state[f"manual_xgb_{feature}"] = float(
                            sample_row.iloc[0][feature]
                        )
                    st.rerun()

        except Exception as e:
            st.error(f"Could not load the FD001 training dataset: {e}")

        values = {}
        input_columns = st.columns(3)

        for index, feature in enumerate(manual_features):
            with input_columns[index % 3]:
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"manual_xgb_{feature}",
                )

        if st.button("Predict RUL with Tuned XGBoost", type="primary"):
            try:
                input_df = pd.DataFrame(
                    [[values[feature] for feature in manual_features]],
                    columns=manual_features,
                )
                prediction = float(manual_xgb.predict(input_df)[0])
                prediction = max(0.0, prediction)

                st.divider()
                result_col1, result_col2 = st.columns(2)
                result_col1.metric(
                    "Predicted Remaining Useful Life", f"{prediction:.2f} cycles"
                )

                if prediction > 60:
                    status = "Healthy"
                    result_col2.success(status)
                elif prediction > 30:
                    status = "Warning"
                    result_col2.warning(status)
                else:
                    status = "Critical"
                    result_col2.error(status)

                st.caption(
                    "Health categories use illustrative project thresholds: "
                    "above 60 = Healthy, above 30 = Warning, and 30 or below = Critical."
                )
            except Exception as e:
                st.error(f"Tuned XGBoost prediction failed: {e}")

    st.divider()
    st.caption("AeroPredict AI | NASA C-MAPSS FD001 | Manual prediction mode")
    st.stop()


# ==================================================
# 9. DATASET UPLOAD
# ==================================================

st.markdown("### Upload Engine Dataset")

uploaded_file = st.file_uploader(
    "Upload NASA C-MAPSS test dataset (.txt)", type=["txt"]
)

if uploaded_file is None:

    st.info("Upload test_FD001.txt to begin engine monitoring.")

    st.markdown("### System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction Model", "LSTM")
    col2.metric("Sequence Length", f"{sequence_length} Cycles")
    col3.metric("Input Features", len(feature_columns))

    st.markdown("### Dashboard Capabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.info("Aircraft engine RUL prediction")
        st.info("Fleet health monitoring")
        st.info("Sensor trend visualization")

    with col2:
        st.info("Machine learning model comparison")
        st.info("SHAP explainability")
        st.info("Downloadable prediction reports")

    st.stop()


# ==================================================
# 9. READ DATASET
# ==================================================

try:

    data = pd.read_csv(uploaded_file, sep=r"\s+", header=None)

    if data.shape[1] != 26:
        st.error("Invalid dataset. Expected 26 columns " "from NASA C-MAPSS.")
        st.stop()

    data.columns = columns

    data = data.sort_values(["Engine_ID", "Cycle"])

    if data.isnull().values.any():
        st.error("Dataset contains missing values.")
        st.stop()

except Exception as e:

    st.error(f"Dataset loading error: {e}")
    st.stop()


# ==================================================
# 10. VALIDATE DATA
# ==================================================

missing_features = [col for col in feature_columns if col not in data.columns]

if missing_features:

    st.error(f"Missing required features: {missing_features}")
    st.stop()

if data.empty:
    st.error("Uploaded dataset is empty.")
    st.stop()

engine_ids = sorted(data["Engine_ID"].unique())


# ==================================================
# 11. GENERATE LSTM PREDICTIONS
# ==================================================


@st.cache_data(show_spinner="Predicting engine health...")
def generate_predictions(data, _model, _scaler):

    predictions = {}

    for engine_id in sorted(data["Engine_ID"].unique()):

        engine_data = data[data["Engine_ID"] == engine_id].sort_values("Cycle")

        if len(engine_data) < sequence_length:
            continue

        last_cycles = engine_data.tail(sequence_length)

        features = last_cycles[feature_columns].values

        scaled_features = _scaler.transform(features)

        sequence = scaled_features.reshape(1, sequence_length, len(feature_columns))

        prediction = _model.predict(sequence, verbose=0)[0][0]

        predictions[engine_id] = max(0, float(prediction))

    return predictions


try:

    predictions = generate_predictions(data, lstm_model, scaler)

except Exception as e:

    st.error(f"Prediction error: {e}")
    st.stop()


if not predictions:

    st.error(f"No engine has at least {sequence_length} cycles.")
    st.stop()


# ==================================================
# 12. FLEET HEALTH CLASSIFICATION
# ==================================================


def get_health_status(rul):

    if rul > 60:
        return "Healthy"

    elif rul > 30:
        return "Warning"

    else:
        return "Critical"


fleet_df = pd.DataFrame(
    {"Engine_ID": list(predictions.keys()), "Predicted_RUL": list(predictions.values())}
)

fleet_df["Health_Status"] = fleet_df["Predicted_RUL"].apply(get_health_status)


healthy_count = (fleet_df["Health_Status"] == "Healthy").sum()

warning_count = (fleet_df["Health_Status"] == "Warning").sum()

critical_count = (fleet_df["Health_Status"] == "Critical").sum()


# ==================================================
# 13. ENGINE SELECTION
# ==================================================

st.divider()

selected_engine = st.selectbox("Select Aircraft Engine", list(predictions.keys()))

engine_data = data[data["Engine_ID"] == selected_engine].sort_values("Cycle")

predicted_rul = predictions[selected_engine]

current_cycle = int(engine_data["Cycle"].max())

health_status = get_health_status(predicted_rul)


# ==================================================
# 14. DASHBOARD PAGE
# ==================================================


# ==================================================
# 14. DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    # Header
    st.markdown("## Fleet Intelligence Overview")

    st.caption(
        "Real-time predictive health insights "
        "from uploaded aircraft engine histories."
    )

    # ----------------------------------------------
    # FLEET KPI CARDS
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "TOTAL ENGINES",
        len(predictions),
        help="Number of engines successfully analyzed",
    )

    col2.metric("HEALTHY ENGINES", int(healthy_count))

    col3.metric("WARNING ENGINES", int(warning_count))

    col4.metric("CRITICAL ENGINES", int(critical_count))

    st.write("")
    st.divider()

    # ----------------------------------------------
    # SELECTED ENGINE OVERVIEW
    # ----------------------------------------------

    st.markdown("### Selected Engine Intelligence")

    left, right = st.columns([1.5, 1])

    with left:

        st.markdown(f"### Engine #{selected_engine}")

        st.caption(f"Latest recorded operating cycle: {current_cycle}")

        st.metric("Estimated Remaining Useful Life", f"{predicted_rul:.1f} cycles")

        if health_status == "Healthy":
            st.success("HEALTH STATUS: HEALTHY")

        elif health_status == "Warning":
            st.warning("HEALTH STATUS: WARNING")

        else:
            st.error("HEALTH STATUS: CRITICAL")

        st.caption(
            "Status thresholds are illustrative and "
            "not certified aviation maintenance limits."
        )

    with right:

        # RUL indicator

        gauge_fig = px.pie(
            names=["Remaining RUL", "Reference"],
            values=[min(predicted_rul, 125), max(0, 125 - min(predicted_rul, 125))],
            hole=0.78,
            color_discrete_sequence=["#38bdf8", "#263750"],
        )

        gauge_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[
                dict(
                    text=f"{predicted_rul:.0f}",
                    x=0.5,
                    y=0.5,
                    font_size=30,
                    font_color="#f8fafc",
                    showarrow=False,
                )
            ],
        )

        st.plotly_chart(gauge_fig, use_container_width=True)

    st.divider()

    # ----------------------------------------------
    # FLEET RUL DISTRIBUTION
    # ----------------------------------------------

    st.markdown("### Fleet RUL Analytics")

    col1, col2 = st.columns([1.7, 1])

    with col1:

        fleet_fig = px.bar(
            fleet_df.sort_values("Engine_ID"),
            x="Engine_ID",
            y="Predicted_RUL",
            color="Health_Status",
            color_discrete_map={
                "Healthy": "#22c55e",
                "Warning": "#f59e0b",
                "Critical": "#ef4444",
            },
            title="Remaining Useful Life by Engine",
            labels={
                "Engine_ID": "Engine ID",
                "Predicted_RUL": "Predicted RUL (Cycles)",
            },
        )

        fleet_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title="Health Status",
            margin=dict(t=50, b=20),
        )

        st.plotly_chart(fleet_fig, use_container_width=True)

    with col2:

        health_counts = pd.DataFrame(
            {
                "Health Status": ["Healthy", "Warning", "Critical"],
                "Engine Count": [
                    int(healthy_count),
                    int(warning_count),
                    int(critical_count),
                ],
            }
        )

        health_fig = px.pie(
            health_counts,
            names="Health Status",
            values="Engine Count",
            hole=0.65,
            color="Health Status",
            color_discrete_map={
                "Healthy": "#22c55e",
                "Warning": "#f59e0b",
                "Critical": "#ef4444",
            },
            title="Fleet Health Distribution",
        )

        health_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20),
        )

        st.plotly_chart(health_fig, use_container_width=True)

    st.divider()

    # ----------------------------------------------
    # ENGINE PREDICTION TABLE
    # ----------------------------------------------

    st.markdown("### Engine Intelligence Report")

    display_df = fleet_df.rename(
        columns={
            "Engine_ID": "Engine ID",
            "Predicted_RUL": "Predicted RUL",
            "Health_Status": "Health Status",
        }
    )

    st.dataframe(
        display_df.style.format({"Predicted RUL": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # ----------------------------------------------
    # DOWNLOAD REPORT
    # ----------------------------------------------

    st.download_button(
        label="Download Fleet Prediction Report",
        data=display_df.to_csv(index=False),
        file_name="AeroPredict_Fleet_Report.csv",
        mime="text/csv",
    )


# ==================================================
# 15. ENGINE MONITORING PAGE
# ==================================================


elif page == "Engine Monitoring":

    st.markdown("## Engine Monitoring")
    st.markdown(
        "Detailed operational analysis and sensor-level monitoring "
        "for individual aircraft engines."
    )

    st.write("")

    # Engine overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Engine ID", f"#{selected_engine}")

    with col2:
        st.metric("Current Cycle", int(current_cycle))

    with col3:
        st.metric("Predicted RUL", f"{predicted_rul:.1f} cycles")

    with col4:
        st.metric("Health Status", health_status)

    st.divider()

    # Sensor selection
    st.markdown("### Sensor Analysis")

    sensor_columns = [col for col in engine_data.columns if col.startswith("Sensor_")]

    if sensor_columns:

        selected_sensor = st.selectbox("Select sensor to analyse", sensor_columns)

        # Sensor trend chart
        fig = px.line(
            engine_data,
            x="Cycle",
            y=selected_sensor,
            markers=True,
            title=f"{selected_sensor} - Operational Trend",
            template="plotly_dark",
        )

        fig.update_traces(line=dict(width=2), marker=dict(size=4))

        fig.update_layout(
            height=430,
            xaxis_title="Operating Cycle",
            yaxis_title="Sensor Reading",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Sensor statistics
        st.markdown("### Sensor Statistics")

        sensor_values = engine_data[selected_sensor]

        stat1, stat2, stat3, stat4 = st.columns(4)

        stat1.metric("Latest Reading", f"{sensor_values.iloc[-1]:.3f}")
        stat2.metric("Average", f"{sensor_values.mean():.3f}")
        stat3.metric("Minimum", f"{sensor_values.min():.3f}")
        stat4.metric("Maximum", f"{sensor_values.max():.3f}")

    else:
        st.warning("No sensor columns found for this engine.")

    st.divider()

    # Engine operational data
    st.markdown("### Engine Operational Data")

    display_columns = [
        "Engine_ID",
        "Cycle",
        "Setting_1",
        "Setting_2",
        "Setting_3",
    ] + sensor_columns

    display_columns = [col for col in display_columns if col in engine_data.columns]

    st.dataframe(
        engine_data[display_columns].sort_values("Cycle", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # Download selected engine data
    csv_data = engine_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Engine Data",
        data=csv_data,
        file_name=f"engine_{selected_engine}_monitoring.csv",
        mime="text/csv",
    )


# ==================================================
# 16. MODEL COMPARISON PAGE
# ==================================================


elif page == "Model Comparison":

    st.markdown("## Model Performance Analysis")
    st.markdown(
        "Compare the predictive performance of different machine learning "
        "and deep learning models on the NASA C-MAPSS FD001 test dataset."
    )

    st.write("")

    comparison_path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")

    if os.path.exists(comparison_path):

        comparison_df = pd.read_csv(comparison_path)

        # Identify best model using the lowest MAE
        best_model = comparison_df.loc[comparison_df["MAE"].idxmin()]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Model", str(best_model["Model"]))

        with col2:
            st.metric("Lowest MAE", f"{best_model['MAE']:.2f}")

        with col3:
            st.metric("Lowest RMSE", f"{comparison_df['RMSE'].min():.2f}")

        st.divider()

        # MAE comparison
        st.markdown("### Mean Absolute Error (MAE)")

        mae_fig = px.bar(
            comparison_df.sort_values("MAE"),
            x="Model",
            y="MAE",
            color="MAE",
            text_auto=".2f",
            title="Model Error Comparison",
            template="plotly_dark",
        )

        mae_fig.update_layout(height=400)
        st.plotly_chart(mae_fig, use_container_width=True)

        # RMSE comparison
        st.markdown("### Root Mean Squared Error (RMSE)")

        rmse_fig = px.bar(
            comparison_df.sort_values("RMSE"),
            x="Model",
            y="RMSE",
            color="RMSE",
            text_auto=".2f",
            title="RMSE Comparison",
            template="plotly_dark",
        )

        rmse_fig.update_layout(height=400)
        st.plotly_chart(rmse_fig, use_container_width=True)

        # R2 comparison
        st.markdown("### R² Score Comparison")

        r2_fig = px.bar(
            comparison_df.sort_values("R2 Score", ascending=False),
            x="Model",
            y="R2 Score",
            color="R2 Score",
            text_auto=".3f",
            title="Model Prediction Accuracy Comparison",
            template="plotly_dark",
        )

        r2_fig.update_layout(height=400)
        st.plotly_chart(r2_fig, use_container_width=True)

        st.divider()

        # Detailed results
        st.markdown("### Complete Model Performance")

        ranked_df = comparison_df.sort_values("MAE").reset_index(drop=True)

        ranked_df.insert(0, "Rank", range(1, len(ranked_df) + 1))

        st.dataframe(ranked_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Model Comparison",
            data=ranked_df.to_csv(index=False).encode("utf-8"),
            file_name="aeropredict_model_comparison.csv",
            mime="text/csv",
        )

        st.info(
            f"{best_model['Model']} achieved the lowest MAE "
            f"on the test dataset. Lower MAE and RMSE indicate "
            f"smaller prediction errors, while higher R² indicates "
            f"better explained variation."
        )

    else:
        st.warning("Model comparison report was not found.")


# ==================================================
# 17. EXPLAINABLE AI PAGE
# ==================================================


elif page == "Explainable AI":

    st.markdown("## Explainable AI")
    st.markdown(
        "Understand which engine operating parameters influence "
        "the Remaining Useful Life predictions."
    )

    st.write("")

    shap_path = os.path.join(BASE_DIR, "reports", "shap_feature_importance.csv")

    if os.path.exists(shap_path):

        shap_df = pd.read_csv(shap_path)

        shap_df = shap_df.sort_values("Importance", ascending=False).reset_index(
            drop=True
        )

        shap_df.insert(0, "Rank", range(1, len(shap_df) + 1))

        top_feature = shap_df.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Features Analysed", len(shap_df))

        with col2:
            st.metric("Most Influential Feature", top_feature["Feature"])

        with col3:
            st.metric("Top Feature Importance", f"{top_feature['Importance']:.4f}")

        st.divider()

        # Feature importance chart
        st.markdown("### Global Feature Importance")

        top_10 = shap_df.head(10).sort_values("Importance", ascending=True)

        fig = px.bar(
            top_10,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            text_auto=".4f",
            title="Top 10 Features Influencing RUL Predictions",
            template="plotly_dark",
        )

        fig.update_layout(
            height=500,
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Engine Feature",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Feature ranking
        st.markdown("### Feature Importance Ranking")

        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download SHAP Report",
            data=shap_df.to_csv(index=False).encode("utf-8"),
            file_name="aeropredict_shap_report.csv",
            mime="text/csv",
        )

        st.divider()

        # Explanation
        st.markdown("### Understanding SHAP")

        st.write(
            "SHAP (SHapley Additive exPlanations) is an "
            "explainability technique used to estimate how "
            "individual input features contribute to model predictions."
        )

        st.info(
            "A higher mean absolute SHAP value indicates greater "
            "influence on the model output in the analysed samples. "
            "It does not establish causation."
        )

        st.warning(
            "This project's SHAP analysis uses a small sample of "
            "test sequences and should be interpreted as preliminary, "
            "not as a definitive ranking of sensor importance."
        )

    else:
        st.warning("SHAP feature importance report was not found.")


# ==================================================
# 18. MODEL INFORMATION PAGE
# ==================================================


elif page == "Model Information":

    st.markdown("## Model Information")
    st.markdown(
        "Technical overview of the AeroPredict AI predictive " "maintenance system."
    )

    st.divider()

    # Project overview
    st.markdown("### Project Overview")

    st.write(
        "AeroPredict AI is a deep learning-based predictive "
        "maintenance system that estimates the Remaining Useful Life "
        "(RUL) of aircraft engines using multivariate sensor data."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset", "NASA C-MAPSS")
    col2.metric("Dataset Subset", "FD001")
    col3.metric("Selected Model", "LSTM")

    st.divider()

    # Model architecture
    st.markdown("### Selected Model Architecture")

    st.markdown("""
    **Long Short-Term Memory (LSTM)**

    - Input: 30 consecutive engine operating cycles
    - Features: 18 selected operational features
    - Output: Predicted Remaining Useful Life (RUL)
    - Architecture: Sequence-based deep learning model
    """)

    st.divider()

    # Test performance
    st.markdown("### Final Test Performance")

    metrics = {"MAE": 16.432417, "RMSE": 23.576680, "R2 Score": 0.678111}

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{metrics['MAE']:.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col3.metric("R² Score", f"{metrics['R2 Score']:.3f}")

    st.caption("Metrics are based on the 100-engine NASA C-MAPSS FD001 " "test set.")

    st.divider()

    # Technology stack
    st.markdown("### Technology Stack")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        **Machine Learning**
        - Python
        - TensorFlow / Keras
        - Scikit-learn
        - SHAP
        - NumPy and Pandas
        """)

    with tech_col2:
        st.markdown("""
        **Application and Visualization**
        - Streamlit
        - Plotly
        - Joblib
        - NASA C-MAPSS Dataset
        """)

    st.divider()

    # Prediction workflow
    st.markdown("### Prediction Workflow")

    st.markdown("""
    1. **Data Input:** Upload aircraft engine sensor data.
    2. **Preprocessing:** Validate columns and organize engine cycles.
    3. **Feature Scaling:** Apply the saved training scaler.
    4. **Sequence Generation:** Extract the latest 30 operating cycles.
    5. **LSTM Prediction:** Estimate the engine's remaining useful life.
    6. **Health Assessment:** Classify the engine based on predicted RUL.
    7. **Reporting:** Display predictions and export monitoring results.
    """)

    st.divider()

    st.success(
        "AeroPredict AI combines time-series deep learning, "
        "explainable AI, and interactive visualization for "
        "predictive aircraft engine maintenance."
    )
