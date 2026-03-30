# 1000409-Mohammed-Saif-ATM
# 🏧 ATM Intelligence – AI Powered Demand Forecasting System

## 📌 Project Overview

ATM Intelligence is an AI-based system designed to predict ATM cash withdrawal demand, detect unusual spikes, and optimize cash refill strategies. The system uses machine learning techniques to improve ATM management efficiency and reduce cash-out risks.

---

## 🚀 Features

* 📊 Demand Forecasting using Machine Learning
* 🚨 Anomaly Detection (Demand Spikes)
* 💰 Smart Cash Refill Recommendation
* ⚠ Risk Alert System (Cash-out prediction)
* 📈 Interactive Visualization Dashboard
* 🌐 Web Application using Streamlit

---

## 🧠 Technologies Used

* Python
* Pandas & NumPy (Data Processing)
* Matplotlib (Visualization)
* Scikit-learn (Machine Learning)
* Streamlit (Web Application)

---

## 📂 Dataset Requirements

The dataset must be in CSV format and include the following columns:

* Date
* Withdrawals
* Location_Type
* Time_of_Day
* Temperature
* Holiday_Flag

---

## ⚙️ Installation & Setup

### 1. Install Required Libraries

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

### 2. Run the Application

```bash
streamlit run your_file_name.py
```

---

## 📊 How It Works

### 🔹 Data Preprocessing

* Converts date into useful features (month, day, week, etc.)
* Handles missing values
* Encodes categorical variables
* Normalizes numerical data

### 🔹 Demand Prediction

* Uses Random Forest Regression to predict ATM withdrawals

### 🔹 Anomaly Detection

* Uses Isolation Forest to detect unusual withdrawal spikes

### 🔹 Refill Optimization

* Recommends refill when predicted demand exceeds 80% of ATM capacity

### 🔹 Risk Detection

* Flags high-risk days when predicted demand exceeds ATM capacity

---

## 📈 Output

* Model performance metrics (MAE, R² Score)
* Demand spike alerts
* High-risk cash-out days
* Forecast graph
* Downloadable results file

---

## 💡 Applications

* Banking & ATM Management
* Financial Planning
* Smart City Infrastructure
* Cash Logistics Optimization

---

## 👨‍💻 Author

**N Mohammed Saif**

---

## 📚 References

* Python Documentation
* Scikit-learn Documentation
* Streamlit Documentation
* Research papers on Random Forest and Isolation Forest

---

## ✅ Conclusion

This project demonstrates how AI and machine learning can be applied in real-world banking systems to improve efficiency, reduce risks, and automate decision-making processes.

---
