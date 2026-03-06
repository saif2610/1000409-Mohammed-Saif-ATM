"""
ATM INTELLIGENCE – AI POWERED DEMAND FORECASTING SYSTEM
Created by: N Mohammed Saif
Advanced Version with Forecasting + Anomaly Detection + Refill Optimization
"""

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("atm_withdrawal_data.csv")

# ===============================
# DATA PREPROCESSING
# ===============================

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Feature Extraction
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Day_of_Week"] = df["Date"].dt.dayofweek
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)

# Weekend Feature
df["Is_Weekend"] = df["Day_of_Week"].apply(lambda x: 1 if x >= 5 else 0)

# Salary Day Feature
df["Is_Salary_Day"] = df["Day"].apply(lambda x: 1 if x in [1, 30] else 0)

# Handle Missing Values
df = df.ffill()

# ===============================
# ENCODING CATEGORICAL VARIABLES
# ===============================
label_cols = ["Location_Type", "Time_of_Day"]

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ===============================
# NORMALIZATION
# ===============================
scaler = MinMaxScaler()

numeric_cols = ["Temperature", "Holiday_Flag"]

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ===============================
# MODEL TRAINING – DEMAND FORECAST
# ===============================

X = df.drop(columns=["Withdrawals", "Date"])
y = df["Withdrawals"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n📊 MODEL PERFORMANCE")
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# ===============================
# DEMAND SPIKE DETECTION
# ===============================

iso_model = IsolationForest(
    contamination=0.05,
    random_state=42
)

numeric_X = X.select_dtypes(include=[np.number])

df["Anomaly"] = iso_model.fit_predict(numeric_X)

df["Demand_Spike"] = df["Anomaly"].apply(
    lambda x: 1 if x == -1 else 0
)

print("\n🚨 Total Demand Spikes Detected:", df["Demand_Spike"].sum())

# ===============================
# SMART CASH REFILL RECOMMENDER
# ===============================

ATM_CAPACITY = 200000

df["Predicted_Demand"] = model.predict(X[X_train.columns])

df["Recommended_Refill"] = np.where(
    df["Predicted_Demand"] > ATM_CAPACITY * 0.8,
    ATM_CAPACITY - df["Predicted_Demand"],
    0
)

# ===============================
# VISUALIZATION
# ===============================

plt.figure(figsize=(12,6))

plt.plot(df["Date"], df["Withdrawals"], label="Actual Withdrawals")
plt.plot(df["Date"], df["Predicted_Demand"], label="Predicted Demand")

plt.title("ATM Demand Forecasting")
plt.xlabel("Date")
plt.ylabel("Withdrawals")

plt.legend()
plt.show()

# ===============================
# RISK ALERT SYSTEM
# ===============================

df["Cashout_Risk"] = np.where(
    df["Predicted_Demand"] > ATM_CAPACITY,
    "HIGH",
    "SAFE"
)

print("\n⚠ High Risk Days:")
print(df[df["Cashout_Risk"] == "HIGH"][["Date", "Predicted_Demand"]])
