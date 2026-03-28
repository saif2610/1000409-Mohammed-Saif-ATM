"""
ATM INTELLIGENCE – AI POWERED DEMAND FORECASTING SYSTEM
Created by: N Mohammed Saif
Final Professional Version (Stable + All Features)
"""

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

print("🚀 Starting ATM Intelligence System...")

# ===============================
# LOAD DATASET (SAFE)
# ===============================
try:
    df = pd.read_csv("atm_withdrawal_data.csv")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ ERROR: Dataset not found or invalid!")
    print(e)
    exit()

# ===============================
# BASIC CHECK
# ===============================
print("\n📌 Columns in Dataset:", df.columns)
print(df.head())

# ===============================
# DATA PREPROCESSING
# ===============================
try:
    df["Date"] = pd.to_datetime(df["Date"])
except:
    print("❌ 'Date' column missing or wrong format")
    exit()

# Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Day_of_Week"] = df["Date"].dt.dayofweek
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)

# Extra Features
df["Is_Weekend"] = df["Day_of_Week"].apply(lambda x: 1 if x >= 5 else 0)
df["Is_Salary_Day"] = df["Day"].apply(lambda x: 1 if x in [1, 30] else 0)

# Fill Missing Values
df = df.ffill()

# ===============================
# ENCODING
# ===============================
label_cols = ["Location_Type", "Time_of_Day"]

for col in label_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    else:
        print(f"⚠ Column '{col}' missing")

# ===============================
# NORMALIZATION
# ===============================
scaler = MinMaxScaler()

numeric_cols = ["Temperature", "Holiday_Flag"]

for col in numeric_cols:
    if col not in df.columns:
        df[col] = 0

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ===============================
# MODEL TRAINING
# ===============================
if "Withdrawals" not in df.columns:
    print("❌ 'Withdrawals' column missing!")
    exit()

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
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# ===============================
# ANOMALY DETECTION
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

print("\n🚨 Demand Spikes Detected:", df["Demand_Spike"].sum())

# ===============================
# SMART REFILL SYSTEM
# ===============================
ATM_CAPACITY = 200000

df["Predicted_Demand"] = model.predict(X)

df["Recommended_Refill"] = np.where(
    df["Predicted_Demand"] > ATM_CAPACITY * 0.8,
    ATM_CAPACITY - df["Predicted_Demand"],
    0
)

# ===============================
# RISK ALERT SYSTEM
# ===============================
df["Cashout_Risk"] = np.where(
    df["Predicted_Demand"] > ATM_CAPACITY,
    "HIGH",
    "SAFE"
)

print("\n⚠ HIGH RISK DAYS:")
print(df[df["Cashout_Risk"] == "HIGH"][["Date", "Predicted_Demand"]])

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
plt.grid()

plt.savefig("forecast.png")  # safer than show
plt.show()

# ===============================
# SAVE OUTPUT FILE
# ===============================
df.to_csv("atm_results_output.csv", index=False)

print("\n💾 Results saved as 'atm_results_output.csv'")
print("📈 Graph saved as 'forecast.png'")

print("\n✅ SYSTEM COMPLETED SUCCESSFULLY!")

input("\nPress Enter to exit...")
