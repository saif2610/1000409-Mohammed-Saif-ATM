# ===============================
# ATM INTELLIGENCE – STREAMLIT APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="ATM Intelligence", layout="wide")

st.title("🏧 ATM Intelligence System")
st.markdown("AI Powered Demand Forecasting + Refill Optimization")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("📂 Upload ATM Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    # ===============================
    # LOAD DATA
    # ===============================
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # ===============================
    # CHECK REQUIRED COLUMNS
    # ===============================
    required_cols = ["Date", "Withdrawals"]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Missing required column: {col}")
            st.stop()

    # ===============================
    # PREPROCESSING
    # ===============================
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)

    df["Is_Weekend"] = df["Day_of_Week"].apply(lambda x: 1 if x >= 5 else 0)
    df["Is_Salary_Day"] = df["Day"].apply(lambda x: 1 if x in [1, 30] else 0)

    df = df.ffill()

    # ===============================
    # ENCODING
    # ===============================
    for col in ["Location_Type", "Time_of_Day"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            df[col] = 0

    # ===============================
    # NORMALIZATION
    # ===============================
    scaler = MinMaxScaler()

    for col in ["Temperature", "Holiday_Flag"]:
        if col not in df.columns:
            df[col] = 0

    df[["Temperature", "Holiday_Flag"]] = scaler.fit_transform(
        df[["Temperature", "Holiday_Flag"]]
    )

    # ===============================
    # MODEL TRAINING
    # ===============================
    X = df.drop(columns=["Withdrawals", "Date"])
    y = df["Withdrawals"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)

    col1.metric("MAE", round(mean_absolute_error(y_test, predictions), 2))
    col2.metric("R² Score", round(r2_score(y_test, predictions), 2))

    # ===============================
    # ANOMALY DETECTION
    # ===============================
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = iso_model.fit_predict(X)

    df["Demand_Spike"] = df["Anomaly"].apply(lambda x: 1 if x == -1 else 0)

    st.subheader("🚨 Demand Spike Detection")
    st.write("Total Spikes Detected:", int(df["Demand_Spike"].sum()))

    # ===============================
    # REFILL SYSTEM
    # ===============================
    ATM_CAPACITY = 200000

    df["Predicted_Demand"] = model.predict(X)

    df["Recommended_Refill"] = np.where(
        df["Predicted_Demand"] > ATM_CAPACITY * 0.8,
        ATM_CAPACITY - df["Predicted_Demand"],
        0
    )

    # ===============================
    # RISK ALERT
    # ===============================
    df["Cashout_Risk"] = np.where(
        df["Predicted_Demand"] > ATM_CAPACITY,
        "HIGH",
        "SAFE"
    )

    st.subheader("⚠ High Risk Days")
    high_risk = df[df["Cashout_Risk"] == "HIGH"][["Date", "Predicted_Demand"]]

    if len(high_risk) > 0:
        st.dataframe(high_risk)
    else:
        st.success("No high-risk days detected")

    # ===============================
    # VISUALIZATION
    # ===============================
    st.subheader("📊 ATM Demand Forecast")

    fig, ax = plt.subplots()

    ax.plot(df["Date"], df["Withdrawals"], label="Actual")
    ax.plot(df["Date"], df["Predicted_Demand"], label="Predicted")

    ax.set_xlabel("Date")
    ax.set_ylabel("Withdrawals")
    ax.legend()

    st.pyplot(fig)

    # ===============================
    # DOWNLOAD RESULTS
    # ===============================
    st.subheader("📥 Download Results")

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="atm_results.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload your ATM dataset to begin")
