# =====================================================
# ATM INTELLIGENCE - FA-2 ASSIGNMENT
# =====================================================
# Course: Data Mining - Artificial Intelligence
# Assignment: ATM Intelligence Demand Forecasting with Data Mining
# Description: Interactive Python script for EDA, Clustering, 
#              and Anomaly Detection on ATM data
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from scipy import stats

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="ATM Intelligence - FA-2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .main-header {font-size: 2rem; color: #1E88E5; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #43A047; font-weight: bold;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE AND INTRODUCTION
# =====================================================
st.title("🏧 ATM Intelligence System")
st.markdown("### FA-2: Building Actionable Insights and Interactive Python Script")
st.markdown("**FinTrust Bank Ltd.** - ATM Demand Forecasting with Data Mining")
st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
DATA_FILE = "atm_cash_management_dataset.csv"

if not os.path.exists(DATA_FILE):
    st.error(f"❌ Data file '{DATA_FILE}' not found!")
    st.stop()

df = pd.read_csv(DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'])

# =====================================================
# SIDEBAR - INTERACTIVE FILTERS
# =====================================================
st.sidebar.header("🔍 Interactive Filters")

# Filter by ATM
atm_ids = df['ATM_ID'].unique().tolist()
selected_atms = st.sidebar.multiselect(
    "Select ATM(s)", 
    atm_ids, 
    default=atm_ids
)

# Filter by Location Type
location_types = df['Location_Type'].unique().tolist()
selected_locations = st.sidebar.multiselect(
    "Select Location Type(s)", 
    location_types, 
    default=location_types
)

# Filter by Day of Week
days_of_week = df['Day_of_Week'].unique().tolist()
selected_days = st.sidebar.multiselect(
    "Select Day(s) of Week", 
    days_of_week, 
    default=days_of_week
)

# Filter by Time of Day
times_of_day = df['Time_of_Day'].unique().tolist()
selected_times = st.sidebar.multiselect(
    "Select Time of Day", 
    times_of_day, 
    default=times_of_day
)

# Apply filters
df_filtered = df[
    (df['ATM_ID'].isin(selected_atms)) &
    (df['Location_Type'].isin(selected_locations)) &
    (df['Day_of_Week'].isin(selected_days)) &
    (df['Time_of_Day'].isin(selected_times))
]

st.sidebar.markdown(f"**Filtered Records:** {len(df_filtered)}")

# =====================================================
# DATASET OVERVIEW
# =====================================================
st.header("📊 Dataset Overview")
st.markdown("Understanding the structure and content of our ATM data.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(df_filtered))
with col2:
    st.metric("ATMs Analyzed", df_filtered['ATM_ID'].nunique())
with col3:
    st.metric("Date Range", f"{df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}")
with col4:
    st.metric("Location Types", df_filtered['Location_Type'].nunique())

# Display data preview
with st.expander("📋 View Dataset Preview"):
    st.dataframe(df_filtered.head(10))
    st.markdown("**Column Descriptions:**")
    st.markdown("""
    - **ATM_ID**: Unique identifier for each ATM
    - **Total_Withdrawals**: Daily cash withdrawal amount
    - **Total_Deposits**: Daily cash deposit amount
    - **Location_Type**: Urban, Suburban, Rural, Metropolitan, Business_Hub
    - **Day_of_Week**: Monday through Sunday
    - **Time_of_Day**: Morning, Afternoon, Evening
    - **Holiday_Flag**: 1 if holiday, 0 otherwise
    - **Special_Event_Flag**: 1 if special event, 0 otherwise
    - **Weather_Condition**: Sunny, Cloudy, Rainy
    - **Nearby_Competitor_ATMs**: Yes/No
    - **Previous_Day_Cash_Level**: Cash available previous day
    - **Cash_Demand_Next_Day**: Predicted demand for next day
    - **Temperature**: Temperature in Celsius
    """)

st.markdown("---")

# =====================================================
# STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================
st.header("📈 Stage 3: Exploratory Data Analysis (EDA)")
st.markdown("EDA is the storytelling stage of data mining. It helps explore the dataset visually to uncover trends, relationships, and patterns.")

# -----------------------------------------------------
# 3.1 DISTRIBUTION ANALYSIS
# -----------------------------------------------------
st.subheader("3.1 Distribution Analysis")
st.markdown("Understanding the distribution of withdrawals and deposits across all ATMs.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Histogram of Total Withdrawals**")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_filtered['Total_Withdrawals'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total Withdrawals ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Total Withdrawals', fontsize=12)
    ax.axvline(df_filtered['Total_Withdrawals'].mean(), color='red', linestyle='--', label=f'Mean: ${df_filtered["Total_Withdrawals"].mean():,.0f}')
    ax.legend()
    st.pyplot(fig)
    st.info("**Observation:** The distribution shows the spread of withdrawal amounts. Peaks indicate common withdrawal values, while the red line shows the average.")

with col2:
    st.markdown("**Histogram of Total Deposits**")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_filtered['Total_Deposits'], bins=30, color='teal', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total Deposits ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Total Deposits', fontsize=12)
    ax.axvline(df_filtered['Total_Deposits'].mean(), color='red', linestyle='--', label=f'Mean: ${df_filtered["Total_Deposits"].mean():,.0f}')
    ax.legend()
    st.pyplot(fig)
    st.info("**Observation:** Deposit distribution helps understand cash inflow patterns across ATMs.")

# Box plots for outliers
st.markdown("**Box Plots - Outlier Detection**")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(df_filtered['Total_Withdrawals'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    ax.set_ylabel('Total Withdrawals ($)', fontsize=11)
    ax.set_title('Box Plot - Withdrawals', fontsize=12)
    st.pyplot(fig)
    st.info("**Observation:** Points above the upper whisker represent high-withdrawal outliers that may need special attention.")

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(df_filtered['Total_Deposits'], patch_artist=True)
    bp['boxes'][0].set_facecolor('teal')
    ax.set_ylabel('Total Deposits ($)', fontsize=11)
    ax.set_title('Box Plot - Deposits', fontsize=12)
    st.pyplot(fig)
    st.info("**Observation:** Box plots help identify outliers and the interquartile range of transaction amounts.")

st.markdown("---")

# -----------------------------------------------------
# 3.2 TIME-BASED TRENDS
# -----------------------------------------------------
st.subheader("3.2 Time-Based Trends")
st.markdown("Analyzing withdrawal patterns over time and across different time periods.")

# Line chart of withdrawals over time
st.markdown("**Line Chart - Withdrawals Over Time**")
fig, ax = plt.subplots(figsize=(12, 5))
for atm in df_filtered['ATM_ID'].unique()[:3]:  # Limit to 3 ATMs for clarity
    atm_data = df_filtered[df_filtered['ATM_ID'] == atm].sort_values('Date')
    ax.plot(atm_data['Date'], atm_data['Total_Withdrawals'], label=atm, alpha=0.7, marker='o', markersize=3)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Total Withdrawals ($)', fontsize=11)
ax.set_title('Withdrawal Trends Over Time by ATM', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig)
st.info("**Observation:** Time series visualization reveals seasonal patterns, peaks, and trends in ATM usage over the analysis period.")

# Patterns by Day of Week
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Withdrawals by Day of Week**")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day = df_filtered.groupby('Day_of_Week')['Total_Withdrawals'].mean().reindex(day_order)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA', '#00ACC1', '#FFB300']
    bars = ax.bar(df_day.index, df_day.values, color=colors)
    ax.set_xlabel('Day of Week', fontsize=11)
    ax.set_ylabel('Avg Withdrawals ($)', fontsize=11)
    ax.set_title('Average Withdrawals by Day of Week', fontsize=12)
    ax.axhline(df_day.mean(), color='red', linestyle='--', label='Overall Average')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.info("**Observation:** Weekdays typically show higher withdrawals than weekends. Salary days (1st, 30th) show peaks.")

with col2:
    st.markdown("**Withdrawals by Time of Day**")
    time_order = ['Morning', 'Afternoon', 'Evening']
    df_time = df_filtered.groupby('Time_of_Day')['Total_Withdrawals'].mean().reindex(time_order)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_time = ['#FFA726', '#42A5F5', '#7E57C2']
    bars = ax.bar(df_time.index, df_time.values, color=colors_time)
    ax.set_xlabel('Time of Day', fontsize=11)
    ax.set_ylabel('Avg Withdrawals ($)', fontsize=11)
    ax.set_title('Average Withdrawals by Time of Day', fontsize=12)
    st.pyplot(fig)
    st.info("**Observation:** Morning hours typically see higher withdrawals as people need cash for daily expenses.")

st.markdown("---")

# -----------------------------------------------------
# 3.3 HOLIDAY & EVENT IMPACT
# -----------------------------------------------------
st.subheader("3.3 Holiday & Event Impact")
st.markdown("Analyzing how holidays and special events affect ATM withdrawal patterns.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Withdrawals by Holiday Flag**")
    df_holiday = df_filtered.groupby('Holiday_Flag')['Total_Withdrawals'].agg(['mean', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Normal Day', 'Holiday']
    colors_h = ['#43A047', '#E53935']
    bars = ax.bar(labels, df_holiday['mean'].values, color=colors_h)
    ax.set_xlabel('Day Type', fontsize=11)
    ax.set_ylabel('Avg Withdrawals ($)', fontsize=11)
    ax.set_title('Average Withdrawals: Holiday vs Normal Day', fontsize=12)
    
    # Add count labels
    for bar, count in zip(bars, df_holiday['count'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'n={count}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)
    st.info("**Observation:** Holidays may show lower withdrawals as people tend to stay home or prepare cash in advance.")

with col2:
    st.markdown("**Withdrawals by Special Event Flag**")
    df_event = df_filtered.groupby('Special_Event_Flag')['Total_Withdrawals'].agg(['mean', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['No Event', 'Special Event']
    colors_e = ['#42A5F5', '#FF7043']
    bars = ax.bar(labels, df_event['mean'].values, color=colors_e)
    ax.set_xlabel('Event Status', fontsize=11)
    ax.set_ylabel('Avg Withdrawals ($)', fontsize=11)
    ax.set_title('Average Withdrawals: Event vs Non-Event Day', fontsize=12)
    
    for bar, count in zip(bars, df_event['count'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'n={count}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)
    st.info("**Observation:** Special events (festivals, sports) can cause significant spikes in cash demand.")

st.markdown("---")

# -----------------------------------------------------
# 3.4 EXTERNAL FACTORS
# -----------------------------------------------------
st.subheader("3.4 External Factors Analysis")
st.markdown("Examining how weather conditions and competitor presence affect ATM usage.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Withdrawals by Weather Condition**")
    df_weather = df_filtered.groupby('Weather_Condition')['Total_Withdrawals'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    weather_colors = {'Sunny': '#FFD54F', 'Cloudy': '#90A4AE', 'Rainy': '#42A5F5'}
    colors_w = [weather_colors.get(w, '#999999') for w in df_weather.index]
    
    bp = ax.boxplot([df_filtered[df_filtered['Weather_Condition'] == w]['Total_Withdrawals'].values 
                     for w in df_weather.index], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_w):
        patch.set_facecolor(color)
    ax.set_xticklabels(df_weather.index)
    ax.set_ylabel('Total Withdrawals ($)', fontsize=11)
    ax.set_title('Withdrawals Distribution by Weather', fontsize=12)
    st.pyplot(fig)
    st.info("**Observation:** Rainy weather tends to reduce ATM visits as people prefer to stay indoors.")

with col2:
    st.markdown("**Impact of Nearby Competitor ATMs**")
    df_competitor = df_filtered.groupby('Nearby_Competitor_ATMs')['Total_Withdrawals'].agg(['mean', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = df_competitor.index.tolist()
    colors_c = ['#66BB6A', '#EF5350']
    bars = ax.bar(labels, df_competitor['mean'].values, color=colors_c)
    ax.set_xlabel('Nearby Competitor ATMs', fontsize=11)
    ax.set_ylabel('Avg Withdrawals ($)', fontsize=11)
    ax.set_title('Withdrawals: With vs Without Competitor ATMs', fontsize=12)
    
    for bar, count in zip(bars, df_competitor['count'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'n={count}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)
    st.info("**Observation:** Surprisingly, ATMs with nearby competitors may show similar or higher usage due to location popularity.")

st.markdown("---")

# -----------------------------------------------------
# 3.5 RELATIONSHIP ANALYSIS
# -----------------------------------------------------
st.subheader("3.5 Relationship Analysis")
st.markdown("Exploring relationships between different variables in the dataset.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Scatter Plot: Previous Day Cash Level vs Next Day Demand**")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df_filtered['Previous_Day_Cash_Level'], 
                         df_filtered['Cash_Demand_Next_Day'],
                         c=df_filtered['Total_Withdrawals'], 
                         cmap='viridis', alpha=0.6)
    ax.set_xlabel('Previous Day Cash Level ($)', fontsize=11)
    ax.set_ylabel('Cash Demand Next Day ($)', fontsize=11)
    ax.set_title('Cash Level vs Next Day Demand', fontsize=12)
    plt.colorbar(scatter, label='Withdrawals')
    st.pyplot(fig)
    
    # Calculate correlation
    corr = df_filtered['Previous_Day_Cash_Level'].corr(df_filtered['Cash_Demand_Next_Day'])
    st.info(f"**Observation:** Correlation coefficient: {corr:.3f}. A positive correlation suggests higher cash levels indicate higher expected demand.")

with col2:
    st.markdown("**Correlation Heatmap of Numeric Features**")
    
    # Select numeric columns
    numeric_cols = ['Total_Withdrawals', 'Total_Deposits', 'Holiday_Flag', 
                    'Special_Event_Flag', 'Previous_Day_Cash_Level', 
                    'Cash_Demand_Next_Day', 'Temperature']
    
    df_numeric = df_filtered[numeric_cols]
    corr_matrix = df_numeric.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap of Numeric Features', fontsize=12)
    st.pyplot(fig)
    st.info("**Observation:** Strong correlations help identify key predictors for demand forecasting.")

st.markdown("---")

# =====================================================
# STAGE 4: CLUSTERING ANALYSIS
# =====================================================
st.header("🎯 Stage 4: Clustering Analysis of ATMs")
st.markdown("Grouping ATMs into clusters based on demand behavior for efficient cash management.")

# -----------------------------------------------------
# 4.1 PREPARE DATA FOR CLUSTERING
# -----------------------------------------------------
st.subheader("4.1 Feature Selection and Preparation")

# Aggregate data by ATM for clustering
atm_agg = df_filtered.groupby('ATM_ID').agg({
    'Total_Withdrawals': 'mean',
    'Total_Deposits': 'mean',
    'Location_Type': 'first',
    'Nearby_Competitor_ATMs': 'first'
}).reset_index()

st.markdown("**Aggregated ATM Data for Clustering:**")
st.dataframe(atm_agg)

# Encode categorical variables
le_location = LabelEncoder()
le_competitor = LabelEncoder()

atm_agg['Location_Type_Encoded'] = le_location.fit_transform(atm_agg['Location_Type'])
atm_agg['Competitor_Encoded'] = le_competitor.fit_transform(atm_agg['Nearby_Competitor_ATMs'])

# Select features for clustering
cluster_features = ['Total_Withdrawals', 'Total_Deposits', 'Location_Type_Encoded', 'Competitor_Encoded']
X_cluster = atm_agg[cluster_features].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

st.info("**Note:** Features have been standardized using StandardScaler to ensure equal weight in clustering.")

# -----------------------------------------------------
# 4.2 DETERMINE OPTIMAL NUMBER OF CLUSTERS
# -----------------------------------------------------
st.subheader("4.2 Determining Optimal Number of Clusters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Elbow Method**")
    inertias = []
    K_range = range(1, min(10, len(atm_agg)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax.set_ylabel('Inertia', fontsize=11)
    ax.set_title('Elbow Method for Optimal K', fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.info("**Observation:** The 'elbow' point indicates the optimal number of clusters where adding more clusters provides diminishing returns.")

with col2:
    st.markdown("**Silhouette Score Analysis**")
    silhouette_scores = []
    K_range_sil = range(2, min(10, len(atm_agg)))
    
    for k in K_range_sil:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range_sil, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Silhouette Score vs Number of Clusters', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Mark the best score
    best_k = K_range_sil[np.argmax(silhouette_scores)]
    ax.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
    ax.legend()
    st.pyplot(fig)
    st.info(f"**Observation:** The optimal number of clusters based on Silhouette Score is **{best_k}**.")

# -----------------------------------------------------
# 4.3 APPLY K-MEANS CLUSTERING
# -----------------------------------------------------
st.subheader("4.3 K-Means Clustering Results")

# Use optimal K (default to 3 if not enough data)
optimal_k = st.slider("Select Number of Clusters", 2, min(5, len(atm_agg)), min(3, len(atm_agg)-1))

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
atm_agg['Cluster'] = kmeans.fit_predict(X_scaled)

# Interpret clusters based on average withdrawals
cluster_stats = atm_agg.groupby('Cluster')['Total_Withdrawals'].mean().sort_values(ascending=False)

# Assign meaningful names
cluster_names = {}
name_mapping = {0: 'High-Demand', 1: 'Steady-Demand', 2: 'Low-Demand'}
sorted_clusters = cluster_stats.index.tolist()

for i, cluster in enumerate(sorted_clusters):
    if i == 0:
        cluster_names[cluster] = 'High-Demand'
    elif i == 1:
        cluster_names[cluster] = 'Steady-Demand' if optimal_k >= 3 else 'Moderate-Demand'
    else:
        cluster_names[cluster] = 'Low-Demand'

atm_agg['Cluster_Name'] = atm_agg['Cluster'].map(cluster_names)

# Display cluster assignments
st.markdown("**Cluster Assignments:**")
st.dataframe(atm_agg[['ATM_ID', 'Total_Withdrawals', 'Total_Deposits', 'Location_Type', 'Cluster', 'Cluster_Name']])

# Cluster visualization
st.markdown("**Cluster Visualization:**")
fig, ax = plt.subplots(figsize=(10, 6))
colors_cluster = ['#E53935', '#43A047', '#1E88E5', '#FB8C00', '#8E24AA']

for i, cluster in enumerate(atm_agg['Cluster'].unique()):
    cluster_data = atm_agg[atm_agg['Cluster'] == cluster]
    ax.scatter(cluster_data['Total_Withdrawals'], 
               cluster_data['Total_Deposits'],
               c=colors_cluster[i % len(colors_cluster)],
               label=f'{cluster_names[cluster]} (Cluster {cluster})',
               s=150, alpha=0.7, edgecolors='black')

ax.set_xlabel('Average Total Withdrawals ($)', fontsize=11)
ax.set_ylabel('Average Total Deposits ($)', fontsize=11)
ax.set_title('ATM Clusters by Withdrawals and Deposits', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Cluster interpretation
st.markdown("**Cluster Interpretation:**")
for cluster in atm_agg['Cluster'].unique():
    cluster_data = atm_agg[atm_agg['Cluster'] == cluster]
    st.markdown(f"""
    - **{cluster_names[cluster]} (Cluster {cluster}):**
      - ATMs: {', '.join(cluster_data['ATM_ID'].values)}
      - Avg Withdrawals: ${cluster_data['Total_Withdrawals'].mean():,.0f}
      - Avg Deposits: ${cluster_data['Total_Deposits'].mean():,.0f}
      - Location Types: {', '.join(cluster_data['Location_Type'].unique())}
    """)

st.markdown("---")

# =====================================================
# STAGE 5: ANOMALY DETECTION
# =====================================================
st.header("🚨 Stage 5: Anomaly Detection on Holidays/Events")
st.markdown("Detecting unusual withdrawal patterns that deviate from normal behavior.")

# -----------------------------------------------------
# 5.1 COMPARE HOLIDAY VS NORMAL DAYS
# -----------------------------------------------------
st.subheader("5.1 Holiday vs Normal Day Comparison")

holiday_withdrawals = df_filtered[df_filtered['Holiday_Flag'] == 1]['Total_Withdrawals']
normal_withdrawals = df_filtered[df_filtered['Holiday_Flag'] == 0]['Total_Withdrawals']

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Holiday Avg Withdrawals", f"${holiday_withdrawals.mean():,.0f}")
with col2:
    st.metric("Normal Day Avg Withdrawals", f"${normal_withdrawals.mean():,.0f}")
with col3:
    diff_pct = ((holiday_withdrawals.mean() - normal_withdrawals.mean()) / normal_withdrawals.mean()) * 100
    st.metric("Difference", f"{diff_pct:.1f}%")

# Box plot comparison
fig, ax = plt.subplots(figsize=(8, 5))
data_to_plot = [normal_withdrawals.values, holiday_withdrawals.values]
bp = ax.boxplot(data_to_plot, patch_artist=True)
bp['boxes'][0].set_facecolor('#43A047')
bp['boxes'][1].set_facecolor('#E53935')
ax.set_xticklabels(['Normal Days', 'Holidays'])
ax.set_ylabel('Total Withdrawals ($)', fontsize=11)
ax.set_title('Withdrawal Distribution: Holiday vs Normal Days', fontsize=12)
st.pyplot(fig)

# -----------------------------------------------------
# 5.2 ANOMALY DETECTION METHODS
# -----------------------------------------------------
st.subheader("5.2 Anomaly Detection Methods")

# Select method
anomaly_method = st.radio(
    "Select Anomaly Detection Method:",
    ["Z-Score", "IQR Method", "Isolation Forest"],
    horizontal=True
)

# Prepare data for anomaly detection
X_anomaly = df_filtered[['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level']].copy()

if anomaly_method == "Z-Score":
    st.markdown("**Z-Score Method:** Detects anomalies as points more than 3 standard deviations from the mean.")
    z_scores = np.abs(stats.zscore(X_anomaly))
    threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0)
    anomalies = (z_scores > threshold).any(axis=1)
    df_filtered['Is_Anomaly'] = anomalies.values
    
elif anomaly_method == "IQR Method":
    st.markdown("**IQR Method:** Uses Interquartile Range to detect outliers.")
    Q1 = X_anomaly.quantile(0.25)
    Q3 = X_anomaly.quantile(0.75)
    IQR = Q3 - Q1
    
    multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5)
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    anomalies = ((X_anomaly < lower_bound) | (X_anomaly > upper_bound)).any(axis=1)
    df_filtered['Is_Anomaly'] = anomalies.values
    
else:  # Isolation Forest
    st.markdown("**Isolation Forest:** Machine learning approach for anomaly detection.")
    contamination = st.slider("Contamination Rate", 0.01, 0.20, 0.05)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X_anomaly)
    df_filtered['Is_Anomaly'] = predictions == -1

# Count anomalies
anomaly_count = df_filtered['Is_Anomaly'].sum()
st.metric("Anomalies Detected", anomaly_count)

# -----------------------------------------------------
# 5.3 VISUALIZE ANOMALIES
# -----------------------------------------------------
st.subheader("5.3 Anomaly Visualization")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot normal points
normal_data = df_filtered[~df_filtered['Is_Anomaly']]
ax.scatter(normal_data['Date'], normal_data['Total_Withdrawals'], 
           c='steelblue', label='Normal', alpha=0.6, s=50)

# Plot anomalies
anomaly_data = df_filtered[df_filtered['Is_Anomaly']]
ax.scatter(anomaly_data['Date'], anomaly_data['Total_Withdrawals'], 
           c='red', label='Anomaly', s=100, marker='^', edgecolors='black')

# Highlight holidays
holiday_data = df_filtered[df_filtered['Holiday_Flag'] == 1]
for _, row in holiday_data.iterrows():
    ax.axvline(row['Date'], color='orange', alpha=0.3, linestyle='--')

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Total Withdrawals ($)', fontsize=11)
ax.set_title('Withdrawals with Detected Anomalies', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig)

# Display anomaly details
if anomaly_count > 0:
    st.markdown("**Detected Anomalies:**")
    anomaly_display = anomaly_data[['Date', 'ATM_ID', 'Total_Withdrawals', 'Holiday_Flag', 'Special_Event_Flag', 'Location_Type']]
    st.dataframe(anomaly_display.sort_values('Total_Withdrawals', ascending=False))
    st.info(f"**Observation:** {anomaly_count} anomalies detected. These represent unusual withdrawal patterns that may require investigation.")

st.markdown("---")

# =====================================================
# STAGE 6: INTERACTIVE PLANNER SUMMARY
# =====================================================
st.header("📋 Stage 6: Summary Dashboard")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total ATMs Analyzed", df_filtered['ATM_ID'].nunique())
with col2:
    st.metric("Clusters Identified", optimal_k)
with col3:
    st.metric("Anomalies Detected", anomaly_count)
with col4:
    st.metric("Avg Daily Withdrawals", f"${df_filtered['Total_Withdrawals'].mean():,.0f}")

# Key insights
st.markdown("### Key Insights from Analysis")
st.markdown(f"""
**EDA Insights:**
- Average daily withdrawals: ${df_filtered['Total_Withdrawals'].mean():,.0f}
- Highest withdrawal day: {df_filtered.groupby('Day_of_Week')['Total_Withdrawals'].mean().idxmax()}
- Peak withdrawal time: {df_filtered.groupby('Time_of_Day')['Total_Withdrawals'].mean().idxmax()}
- Holiday impact: {diff_pct:.1f}% difference from normal days

**Clustering Insights:**
- ATMs grouped into {optimal_k} distinct clusters based on demand behavior
- High-demand ATMs: Urban and Metropolitan areas
- Low-demand ATMs: Rural locations

**Anomaly Detection:**
- {anomaly_count} unusual patterns detected using {anomaly_method}
- Anomalies often coincide with holidays and special events
""")

# =====================================================
# DOWNLOAD RESULTS
# =====================================================
st.markdown("---")
st.header("📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    # Download clustered data
    st.download_button(
        label="📊 Download Cluster Results (CSV)",
        data=atm_agg.to_csv(index=False),
        file_name="atm_clusters.csv",
        mime="text/csv"
    )

with col2:
    # Download anomaly data
    st.download_button(
        label="🚨 Download Anomaly Report (CSV)",
        data=df_filtered.to_csv(index=False),
        file_name="atm_anomaly_report.csv",
        mime="text/csv"
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>ATM Intelligence System - FA-2 Assignment</strong></p>
    <p>Course: Data Mining - Artificial Intelligence | FinTrust Bank Ltd.</p>
    <p>Developed with Python, Streamlit, Scikit-learn, and Pandas</p>
</div>
""", unsafe_allow_html=True)
