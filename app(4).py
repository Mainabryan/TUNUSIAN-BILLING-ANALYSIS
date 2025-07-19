import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("STEG_BILLING_HISTORY(1).csv")

data = load_data()

# Title and summary
st.title("Tunisian Electricity Billing Dashboard")
st.markdown("Explore trends in electricity consumption and client billing from 2005 to 2019.")

# Sidebar filters
st.sidebar.header("Filter Options")
tarif = st.sidebar.multiselect("Select Tarif Type", data['tarif_type'].unique())
counter_type = st.sidebar.multiselect("Select Counter Type", data['counter_type'].unique())

filtered_data = data.copy()

if tarif:
    filtered_data = filtered_data[filtered_data['tarif_type'].isin(tarif)]
if counter_type:
    filtered_data = filtered_data[filtered_data['counter_type'].isin(counter_type)]

# Basic stats
st.subheader("Dataset Overview")
st.write(filtered_data.describe())

# Consumption Plot
st.subheader("Electricity Consumption Levels")
consumption_cols = [
    'consommation_level_1',
    'consommation_level_2',
    'consommation_level_3',
    'consommation_level_4'
]

fig, ax = plt.subplots(figsize=(10, 5))
filtered_data[consumption_cols].mean().plot(kind='bar', ax=ax)
plt.title("Average Consumption per Level")
plt.ylabel("kWh")
st.pyplot(fig)

# Index Trend
st.subheader("Old vs New Index")
fig2, ax2 = plt.subplots()
sns.histplot(data=filtered_data, x='old_index', color='blue', label='Old Index', kde=True)
sns.histplot(data=filtered_data, x='new_index', color='green', label='New Index', kde=True)
plt.legend()
plt.title("Distribution of Old vs New Index")
st.pyplot(fig2)

# Footer
st.markdown("Built with ❤️ using Streamlit")
