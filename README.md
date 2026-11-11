# ✈️ AeroPredict AI
### Aircraft Engine Remaining Useful Life (RUL) Prediction Using Deep Learning

AeroPredict AI is a predictive maintenance application that estimates the Remaining Useful Life (RUL) of aircraft engines using multivariate sensor data from the NASA C-MAPSS dataset.

The project uses time-series deep learning to analyse engine operating conditions and predict how many operational cycles remain before an engine reaches its end of useful life.

## 🚀 Key Features

- **RUL Prediction:** Estimates the remaining useful life of aircraft engines using an LSTM model.
- **Fleet Health Monitoring:** Displays engine health classifications and fleet-level analytics.
- **Engine Monitoring:** Interactive sensor trends and operational statistics.
- **Model Comparison:** Compares Random Forest, XGBoost, ANN, LSTM, and GRU models.
- **Explainable AI:** Uses SHAP to investigate feature contributions to predictions.
- **Interactive Dashboard:** Built with Streamlit and Plotly.
- **Data Export:** Download engine monitoring data and model analysis reports.

## 🧠 Model Performance

The following results were obtained on the NASA C-MAPSS FD001 test dataset.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Random Forest | 20.37 | 27.42 | 0.565 |
| Tuned XGBoost | 18.07 | 24.92 | 0.640 |
| ANN | 19.13 | 25.61 | 0.620 |
| **LSTM** | **16.43** | **23.58** | **0.678** |
| GRU | 19.21 | 27.11 | 0.575 |

**Best-performing model:** LSTM

- MAE: 16.43 cycles
- RMSE: 23.58 cycles
- R² Score: 0.678

## 🛠️ Technology Stack

**Programming:** Python

**Deep Learning:** TensorFlow, Keras

**Machine Learning:** Scikit-learn, XGBoost

**Data Processing:** Pandas, NumPy

**Explainable AI:** SHAP

**Visualization:** Plotly, Matplotlib, Seaborn

**Application:** Streamlit

**Model Persistence:** Joblib

## 📂 Project Structure

```text
AeroPredict-AI/
│
├── app/
│   └── app.py
│
├── dataset/
│   └── raw/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── models/
│   ├── aeropredict_lstm.keras
│   └── aeropredict_config.pkl
│
├── notebooks/
│   └── AeroPredict.ipynb
│
├── reports/
│   ├── model_comparison.csv
│   └── shap_feature_importance.csv
│
└── requirements.txt
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AeroPredict-AI.git
cd AeroPredict-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Obtain the NASA C-MAPSS dataset and place the FD001 files inside:

```text
dataset/raw/
```

### 5. Run the application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## 🔬 Methodology

1. Load the NASA C-MAPSS FD001 training and test datasets.
2. Perform data preprocessing and feature selection.
3. Calculate training RUL values using engine cycle information.
4. Split training and validation data by engine ID.
5. Train and evaluate traditional machine learning and deep learning models.
6. Generate time-series sequences using 30 consecutive operating cycles.
7. Evaluate models on the held-out test engines.
8. Apply SHAP to investigate feature contributions.
9. Integrate the selected LSTM model into a Streamlit dashboard.

## 📊 Explainable AI

SHAP is used to estimate the contribution of input features to the LSTM predictions.

The current analysis identifies Sensor_11 as the most influential feature among the analysed samples. This is a preliminary result based on a small sample of test sequences, not a definitive ranking across the entire fleet.

## 🎯 Project Objective

To demonstrate how time-series deep learning and explainable AI can support predictive maintenance by estimating aircraft engine remaining useful life and presenting actionable monitoring insights.

## ⚠️ Disclaimer

This is an academic and demonstration project using the NASA C-MAPSS dataset. Its predictions are not certified for real-world aircraft maintenance or flight-safety decisions.

## 👨‍💻 Author

**Shibin Mathew**

GitHub: https://github.com/Luff12