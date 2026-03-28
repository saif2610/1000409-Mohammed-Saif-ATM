# ===============================
# ATM INTELLIGENCE – STREAMLIT APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
# PRESET DATA FILE PATH
# ===============================
DATA_FILE = "atm_cash_management_dataset.csv"

# Check if data file exists
if not os.path.exists(DATA_FILE):
    st.error(f"❌ Data file '{DATA_FILE}' not found! Please ensure the file is in the same directory.")
    st.stop()

st.success(f"✅ Using preset data file: **{DATA_FILE}**")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_FILE)

st.subheader("📊 Dataset Preview")
st.write(f"**Total Records:** {len(df)}")
st.write(df.head(10))

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
st.subheader("⚙️ Data Preprocessing")

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Day_of_Week"] = df["Date"].dt.dayofweek
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)

df["Is_Weekend"] = df["Day_of_Week"].apply(lambda x: 1 if x >= 5 else 0)
df["Is_Salary_Day"] = df["Day"].apply(lambda x: 1 if x in [1, 30] else 0)

df = df.ffill()

st.info("✅ Date features extracted and missing values filled")

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
st.subheader("🤖 Model Training")

with st.spinner("Training Random Forest model..."):
    X = df.drop(columns=["Withdrawals", "Date"])
    y = df["Withdrawals"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

st.success("✅ Model trained successfully!")

st.subheader("📈 Model Performance")
col1, col2, col3 = st.columns(3)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

col1.metric("Mean Absolute Error", f"{mae:,.2f}")
col2.metric("R² Score", f"{r2:.4f}")
col3.metric("Training Samples", len(X_train))

# ===============================
# ANOMALY DETECTION
# ===============================
st.subheader("🚨 Demand Spike Detection")

with st.spinner("Detecting anomalies..."):
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = iso_model.fit_predict(X)

    df["Demand_Spike"] = df["Anomaly"].apply(lambda x: 1 if x == -1 else 0)

spike_count = int(df["Demand_Spike"].sum())
st.write(f"**Total Spikes Detected:** {spike_count}")

if spike_count > 0:
    spike_dates = df[df["Demand_Spike"] == 1][["Date", "Withdrawals"]]
    st.write("**Spike Dates:**")
    st.dataframe(spike_dates)

# ===============================
# REFILL SYSTEM
# ===============================
st.subheader("💰 Refill Recommendation System")

ATM_CAPACITY = st.number_input("ATM Capacity ($)", value=200000, min_value=50000, max_value=1000000, step=10000)

df["Predicted_Demand"] = model.predict(X)

df["Recommended_Refill"] = np.where(
    df["Predicted_Demand"] > ATM_CAPACITY * 0.8,
    ATM_CAPACITY - df["Predicted_Demand"],
    0
)

# Calculate refill stats
total_refill_needed = df[df["Recommended_Refill"] > 0]["Recommended_Refill"].sum()
days_need_refill = (df["Recommended_Refill"] > 0).sum()

col1, col2 = st.columns(2)
col1.metric("Days Needing Refill", days_need_refill)
col2.metric("Total Refill Amount", f"${total_refill_needed:,.0f}")

# ===============================
# RISK ALERT
# ===============================
st.subheader("⚠️ High Risk Days")

df["Cashout_Risk"] = np.where(
    df["Predicted_Demand"] > ATM_CAPACITY,
    "HIGH",
    "SAFE"
)

high_risk = df[df["Cashout_Risk"] == "HIGH"][["Date", "Predicted_Demand", "Withdrawals"]]

if len(high_risk) > 0:
    st.warning(f"⚠️ {len(high_risk)} high-risk days detected!")
    st.dataframe(high_risk)
else:
    st.success("✅ No high-risk days detected")

# ===============================
# VISUALIZATION
# ===============================
st.subheader("📊 ATM Demand Forecast")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df["Date"], df["Withdrawals"], label="Actual", color="blue", alpha=0.7)
ax.plot(df["Date"], df["Predicted_Demand"], label="Predicted", color="orange", alpha=0.7)

# Mark demand spikes
spike_df = df[df["Demand_Spike"] == 1]
if len(spike_df) > 0:
    ax.scatter(spike_df["Date"], spike_df["Withdrawals"], color="red", s=100, label="Demand Spike", zorder=5, marker="^")

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Withdrawals ($)", fontsize=12)
ax.set_title("Actual vs Predicted ATM Withdrawals", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# ===============================
# FEATURE IMPORTANCE
# ===============================
st.subheader("📊 Feature Importance")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance for Withdrawal Prediction')
ax2.invert_yaxis()

st.pyplot(fig2)

# ===============================
# SUMMARY STATISTICS
# ===============================
st.subheader("📈 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Daily Withdrawals", f"${df['Withdrawals'].mean():,.0f}")
col2.metric("Max Withdrawals", f"${df['Withdrawals'].max():,.0f}")
col3.metric("Min Withdrawals", f"${df['Withdrawals'].min():,.0f}")
col4.metric("Std Deviation", f"${df['Withdrawals'].std():,.0f}")

# ===============================
# DOWNLOAD RESULTS
# ===============================
st.subheader("📥 Download Results")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📄 Download Full Results (CSV)",
        data=df.to_csv(index=False),
        file_name="atm_results.csv",
        mime="text/csv"
    )

with col2:
    # Create summary report
    summary = f"""
    ATM Intelligence Report
    =======================
    
    Data Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
    Total Records: {len(df)}
    
    Model Performance:
    - MAE: {mae:,.2f}
    - R² Score: {r2:.4f}
    
    Demand Analysis:
    - Average Daily Withdrawals: ${df['Withdrawals'].mean():,.0f}
    - Maximum Withdrawals: ${df['Withdrawals'].max():,.0f}
    - Minimum Withdrawals: ${df['Withdrawals'].min():,.0f}
    
    Anomaly Detection:
    - Demand Spikes Detected: {spike_count}
    
    Risk Assessment:
    - High Risk Days: {len(high_risk)}
    - Days Needing Refill: {days_need_refill}
    
    ATM Capacity: ${ATM_CAPACITY:,}
    """
    
    st.download_button(
        label="📋 Download Summary Report (TXT)",
        data=summary,
        file_name="atm_summary_report.txt",
        mime="text/plain"
    )

st.markdown("---")
st.markdown("### ✅ Analysis Complete!")
