# =====================================================
# ATM INTELLIGENCE DASHBOARD
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
    page_title="ATM Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2rem; color: #1E88E5; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #43A047; font-weight: bold;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.title("🏧 ATM Intelligence Dashboard")
st.markdown("**FinTrust Bank Ltd.** - ATM Demand Forecasting & Analytics")
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
# SIDEBAR - FILTERS
# =====================================================
st.sidebar.header("🔍 Filters")

atm_ids = df['ATM_ID'].unique().tolist()
selected_atms = st.sidebar.multiselect("Select ATM(s)", atm_ids, default=atm_ids)

location_types = df['Location_Type'].unique().tolist()
selected_locations = st.sidebar.multiselect("Select Location Type(s)", location_types, default=location_types)

days_of_week = df['Day_of_Week'].unique().tolist()
selected_days = st.sidebar.multiselect("Select Day(s) of Week", days_of_week, default=days_of_week)

times_of_day = df['Time_of_Day'].unique().tolist()
selected_times = st.sidebar.multiselect("Select Time of Day", times_of_day, default=times_of_day)

# Apply filters with reset index to avoid index mismatch issues
df_filtered = df[
    (df['ATM_ID'].isin(selected_atms)) &
    (df['Location_Type'].isin(selected_locations)) &
    (df['Day_of_Week'].isin(selected_days)) &
    (df['Time_of_Day'].isin(selected_times))
].reset_index(drop=True)

st.sidebar.markdown(f"**Filtered Records:** {len(df_filtered)}")

if len(df_filtered) == 0:
    st.warning("⚠️ No data matches the selected filters.")
    st.stop()

# =====================================================
# KEY METRICS
# =====================================================
st.header("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(df_filtered):,}")
with col2:
    st.metric("ATMs Analyzed", df_filtered['ATM_ID'].nunique())
with col3:
    st.metric("Avg Withdrawals", f"${df_filtered['Total_Withdrawals'].mean():,.0f}")
with col4:
    st.metric("Avg Deposits", f"${df_filtered['Total_Deposits'].mean():,.0f}")

with st.expander("📋 View Dataset"):
    st.dataframe(df_filtered.head(10))

st.markdown("---")

# =====================================================
# DISTRIBUTION ANALYSIS
# =====================================================
st.header("📈 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_filtered['Total_Withdrawals'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total Withdrawals ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Total Withdrawals')
    ax.axvline(df_filtered['Total_Withdrawals'].mean(), color='red', linestyle='--', label=f'Mean: ${df_filtered["Total_Withdrawals"].mean():,.0f}')
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_filtered['Total_Deposits'], bins=30, color='teal', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total Deposits ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Total Deposits')
    ax.axvline(df_filtered['Total_Deposits'].mean(), color='red', linestyle='--', label=f'Mean: ${df_filtered["Total_Deposits"].mean():,.0f}')
    ax.legend()
    st.pyplot(fig)

# Box plots
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(df_filtered['Total_Withdrawals'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    ax.set_ylabel('Total Withdrawals ($)')
    ax.set_title('Box Plot - Withdrawals')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(df_filtered['Total_Deposits'], patch_artist=True)
    bp['boxes'][0].set_facecolor('teal')
    ax.set_ylabel('Total Deposits ($)')
    ax.set_title('Box Plot - Deposits')
    st.pyplot(fig)

st.markdown("---")

# =====================================================
# TIME-BASED ANALYSIS
# =====================================================
st.header("📅 Time-Based Analysis")

# Line chart
fig, ax = plt.subplots(figsize=(12, 5))
for atm in df_filtered['ATM_ID'].unique()[:3]:
    atm_data = df_filtered[df_filtered['ATM_ID'] == atm].sort_values('Date')
    ax.plot(atm_data['Date'], atm_data['Total_Withdrawals'], label=atm, alpha=0.7, marker='o', markersize=3)
ax.set_xlabel('Date')
ax.set_ylabel('Total Withdrawals ($)')
ax.set_title('Withdrawal Trends Over Time')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig)

col1, col2 = st.columns(2)

with col1:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day = df_filtered.groupby('Day_of_Week')['Total_Withdrawals'].mean().reindex(day_order)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA', '#00ACC1', '#FFB300']
    ax.bar(df_day.index, df_day.values, color=colors)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Avg Withdrawals ($)')
    ax.set_title('Average Withdrawals by Day of Week')
    ax.axhline(df_day.mean(), color='red', linestyle='--', label='Average')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    time_order = ['Morning', 'Afternoon', 'Evening']
    df_time = df_filtered.groupby('Time_of_Day')['Total_Withdrawals'].mean().reindex(time_order)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_time = ['#FFA726', '#42A5F5', '#7E57C2']
    ax.bar(df_time.index, df_time.values, color=colors_time)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Avg Withdrawals ($)')
    ax.set_title('Average Withdrawals by Time of Day')
    st.pyplot(fig)

st.markdown("---")

# =====================================================
# HOLIDAY & EVENT IMPACT
# =====================================================
st.header("🎉 Holiday & Event Impact")

col1, col2 = st.columns(2)

with col1:
    df_holiday = df_filtered.groupby('Holiday_Flag')['Total_Withdrawals'].agg(['mean', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Normal Day', 'Holiday']
    colors_h = ['#43A047', '#E53935']
    bars = ax.bar(labels, df_holiday['mean'].values, color=colors_h)
    ax.set_xlabel('Day Type')
    ax.set_ylabel('Avg Withdrawals ($)')
    ax.set_title('Withdrawals: Holiday vs Normal Day')
    
    for bar, count in zip(bars, df_holiday['count'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'n={count}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)

with col2:
    df_event = df_filtered.groupby('Special_Event_Flag')['Total_Withdrawals'].agg(['mean', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['No Event', 'Special Event']
    colors_e = ['#42A5F5', '#FF7043']
    bars = ax.bar(labels, df_event['mean'].values, color=colors_e)
    ax.set_xlabel('Event Status')
    ax.set_ylabel('Avg Withdrawals ($)')
    ax.set_title('Withdrawals: Event vs Non-Event Day')
    
    for bar, count in zip(bars, df_event['count'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'n={count}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)

st.markdown("---")

# =====================================================
# EXTERNAL FACTORS
# =====================================================
st.header("🌤️ External Factors")

col1, col2 = st.columns(2)

with col1:
    df_weather = df_filtered.groupby('Weather_Condition')['Total_Withdrawals'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    weather_colors = {'Sunny': '#FFD54F', 'Cloudy': '#90A4AE', 'Rainy': '#42A5F5'}
    colors_w = [weather_colors.get(w, '#999999') for w in df_weather.index]
    
    bp = ax.boxplot([df_filtered[df_filtered['Weather_Condition'] == w]['Total_Withdrawals'].values 
                     for w in df_weather.index], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_w):
        patch.set_facecolor(color)
    ax.set_xticklabels(df_weather.index)
    ax.set_ylabel('Total Withdrawals ($)')
    ax.set_title('Withdrawals by Weather Condition')
    st.pyplot(fig)

with col2:
    df_competitor = df_filtered.groupby('Nearby_Competitor_ATMs')['Total_Withdrawals'].agg(['mean', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = df_competitor.index.tolist()
    colors_c = ['#66BB6A', '#EF5350']
    bars = ax.bar(labels, df_competitor['mean'].values, color=colors_c)
    ax.set_xlabel('Nearby Competitor ATMs')
    ax.set_ylabel('Avg Withdrawals ($)')
    ax.set_title('Impact of Competitor ATMs')
    st.pyplot(fig)

st.markdown("---")

# =====================================================
# CORRELATION ANALYSIS
# =====================================================
st.header("📊 Correlation Analysis")

available_numeric_cols = [col for col in ['Total_Withdrawals', 'Total_Deposits', 'Holiday_Flag', 
                'Special_Event_Flag', 'Previous_Day_Cash_Level', 
                'Cash_Demand_Next_Day', 'Temperature'] if col in df_filtered.columns]

if len(available_numeric_cols) >= 2:
    df_numeric = df_filtered[available_numeric_cols]
    corr_matrix = df_numeric.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

st.markdown("---")

# =====================================================
# CLUSTERING
# =====================================================
st.header("🎯 ATM Clustering")

atm_agg = df_filtered.groupby('ATM_ID').agg({
    'Total_Withdrawals': 'mean',
    'Total_Deposits': 'mean',
    'Location_Type': 'first',
    'Nearby_Competitor_ATMs': 'first'
}).reset_index()

le_location = LabelEncoder()
le_competitor = LabelEncoder()

atm_agg['Location_Type_Encoded'] = le_location.fit_transform(atm_agg['Location_Type'])
atm_agg['Competitor_Encoded'] = le_competitor.fit_transform(atm_agg['Nearby_Competitor_ATMs'])

cluster_features = ['Total_Withdrawals', 'Total_Deposits', 'Location_Type_Encoded', 'Competitor_Encoded']
X_cluster = atm_agg[cluster_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

col1, col2 = st.columns(2)

with col1:
    inertias = []
    K_range = range(1, min(10, len(atm_agg)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    silhouette_scores = []
    K_range_sil = range(2, min(10, len(atm_agg)))
    
    for k in K_range_sil:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range_sil, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score Analysis')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

optimal_k = st.slider("Select Number of Clusters", 2, min(5, len(atm_agg)), min(3, len(atm_agg)-1) if len(atm_agg) > 3 else 2)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
atm_agg['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_stats = atm_agg.groupby('Cluster')['Total_Withdrawals'].mean().sort_values(ascending=False)

cluster_names = {}
sorted_clusters = cluster_stats.index.tolist()

for i, cluster in enumerate(sorted_clusters):
    if i == 0:
        cluster_names[cluster] = 'High-Demand'
    elif i == 1:
        cluster_names[cluster] = 'Steady-Demand' if optimal_k >= 3 else 'Moderate-Demand'
    else:
        cluster_names[cluster] = 'Low-Demand'

atm_agg['Cluster_Name'] = atm_agg['Cluster'].map(cluster_names)

st.dataframe(atm_agg[['ATM_ID', 'Total_Withdrawals', 'Total_Deposits', 'Location_Type', 'Cluster', 'Cluster_Name']])

fig, ax = plt.subplots(figsize=(10, 6))
colors_cluster = ['#E53935', '#43A047', '#1E88E5', '#FB8C00', '#8E24AA']

for i, cluster in enumerate(atm_agg['Cluster'].unique()):
    cluster_data = atm_agg[atm_agg['Cluster'] == cluster]
    ax.scatter(cluster_data['Total_Withdrawals'], 
               cluster_data['Total_Deposits'],
               c=colors_cluster[i % len(colors_cluster)],
               label=f'{cluster_names[cluster]}',
               s=150, alpha=0.7, edgecolors='black')

ax.set_xlabel('Avg Withdrawals ($)')
ax.set_ylabel('Avg Deposits ($)')
ax.set_title('ATM Clusters')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown("---")

# =====================================================
# ANOMALY DETECTION
# =====================================================
st.header("🚨 Anomaly Detection")

holiday_withdrawals = df_filtered[df_filtered['Holiday_Flag'] == 1]['Total_Withdrawals']
normal_withdrawals = df_filtered[df_filtered['Holiday_Flag'] == 0]['Total_Withdrawals']

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Holiday Avg", f"${holiday_withdrawals.mean():,.0f}" if len(holiday_withdrawals) > 0 else "N/A")
with col2:
    st.metric("Normal Day Avg", f"${normal_withdrawals.mean():,.0f}")
with col3:
    if len(holiday_withdrawals) > 0 and len(normal_withdrawals) > 0:
        diff_pct = ((holiday_withdrawals.mean() - normal_withdrawals.mean()) / normal_withdrawals.mean()) * 100
        st.metric("Difference", f"{diff_pct:.1f}%")
    else:
        st.metric("Difference", "N/A")

fig, ax = plt.subplots(figsize=(8, 5))
if len(holiday_withdrawals) > 0 and len(normal_withdrawals) > 0:
    data_to_plot = [normal_withdrawals.values, holiday_withdrawals.values]
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    bp['boxes'][0].set_facecolor('#43A047')
    bp['boxes'][1].set_facecolor('#E53935')
    ax.set_xticklabels(['Normal Days', 'Holidays'])
    ax.set_ylabel('Total Withdrawals ($)')
    ax.set_title('Holiday vs Normal Day Withdrawals')
st.pyplot(fig)

anomaly_method = st.radio("Select Anomaly Detection Method:", ["Z-Score", "IQR Method", "Isolation Forest"], horizontal=True)

X_anomaly = df_filtered[['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level']].copy()
X_anomaly_values = X_anomaly.values

# Initialize anomaly column
anomaly_labels = np.zeros(len(df_filtered), dtype=bool)

if anomaly_method == "Z-Score":
    threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0)
    z_scores = np.abs(stats.zscore(X_anomaly_values))
    anomaly_labels = (z_scores > threshold).any(axis=1)
    
elif anomaly_method == "IQR Method":
    multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5)
    Q1 = X_anomaly.quantile(0.25).values
    Q3 = X_anomaly.quantile(0.75).values
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    anomaly_labels = ((X_anomaly_values < lower_bound) | (X_anomaly_values > upper_bound)).any(axis=1)
    
else:
    contamination = st.slider("Contamination Rate", 0.01, 0.20, 0.05)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X_anomaly_values)
    anomaly_labels = predictions == -1

# Assign to dataframe using numpy array (avoids index mismatch)
df_filtered['Is_Anomaly'] = anomaly_labels

anomaly_count = df_filtered['Is_Anomaly'].sum()
st.metric("Anomalies Detected", anomaly_count)

fig, ax = plt.subplots(figsize=(12, 6))

normal_data = df_filtered[~df_filtered['Is_Anomaly']]
ax.scatter(normal_data['Date'], normal_data['Total_Withdrawals'], 
           c='steelblue', label='Normal', alpha=0.6, s=50)

anomaly_data = df_filtered[df_filtered['Is_Anomaly']]
ax.scatter(anomaly_data['Date'], anomaly_data['Total_Withdrawals'], 
           c='red', label='Anomaly', s=100, marker='^', edgecolors='black')

ax.set_xlabel('Date')
ax.set_ylabel('Total Withdrawals ($)')
ax.set_title('Withdrawals with Anomalies')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig)

if anomaly_count > 0:
    st.subheader("Detected Anomalies")
    st.dataframe(anomaly_data[['Date', 'ATM_ID', 'Total_Withdrawals', 'Holiday_Flag', 'Special_Event_Flag', 'Location_Type']].sort_values('Total_Withdrawals', ascending=False))

st.markdown("---")

# =====================================================
# SUMMARY
# =====================================================
st.header("📋 Summary Dashboard")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("ATMs Analyzed", df_filtered['ATM_ID'].nunique())
with col2:
    st.metric("Clusters", optimal_k)
with col3:
    st.metric("Anomalies", anomaly_count)
with col4:
    st.metric("Avg Daily Withdrawals", f"${df_filtered['Total_Withdrawals'].mean():,.0f}")

st.markdown("---")

# =====================================================
# DOWNLOAD
# =====================================================
st.header("📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📊 Download Cluster Results (CSV)",
        data=atm_agg.to_csv(index=False),
        file_name="atm_clusters.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        label="🚨 Download Anomaly Report (CSV)",
        data=df_filtered.to_csv(index=False),
        file_name="atm_anomaly_report.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>ATM Intelligence Dashboard</strong> | FinTrust Bank Ltd.</p>
</div>
""", unsafe_allow_html=True)
