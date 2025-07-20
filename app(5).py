import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor  # Uncomment to use XGBoost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io
import base64

# Set page config
st.set_page_config(page_title="Tunisian Electricity Billing Analysis", layout="wide")

# Title and description
st.title("📊 Tunisian Electricity Billing Analysis")
st.markdown("""
This app analyzes the STEG billing dataset, performs exploratory data analysis (EDA), 
trains a machine learning model to predict monthly consumption, and allows custom predictions.
""")

# Sidebar for file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload STEG_BILLING_HISTORY.csv", type=["csv"])

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Function to load and preprocess data
def load_and_preprocess_data(file):
    try:
        data = pd.read_csv(file)
        # Feature engineering
        data['monthly_consumption'] = data['new_index'] - data['old_index']
        data['total_consumption'] = (data['consommation_level_1'] + 
                                    data['consommation_level_2'] + 
                                    data['consommation_level_3'] + 
                                    data['consommation_level_4'])
        # Drop redundant columns
        columns_to_drop = ['consommation_level_1', 'consommation_level_2', 
                          'consommation_level_3', 'consommation_level_4', 
                          'old_index', 'new_index']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Load data if file is uploaded
if uploaded_file:
    st.session_state.data = load_and_preprocess_data(uploaded_file)
    st.success("Dataset loaded successfully!")

# EDA Section
if st.session_state.data is not None:
    data = st.session_state.data
    with st.expander("📋 Dataset Overview", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape of Dataset**")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            st.write("**Data Types**")
            st.write(data.dtypes)
        with col2:
            st.write("**Missing Values**")
            st.write(data.isnull().sum())
            st.write("**First 5 Rows**")
            st.dataframe(data.head())

    with st.expander("📈 Exploratory Data Analysis"):
        st.subheader("Consumption Trends")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(data['monthly_consumption'], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of Monthly Consumption")
            ax.set_xlabel("Monthly Consumption")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(y=data['monthly_consumption'], ax=ax)
            ax.set_title("Boxplot of Monthly Consumption")
            ax.set_ylabel("Monthly Consumption")
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        fig, ax = plt.subplots()
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    # Model Training Section
    with st.expander("🧠 Model Training"):
        st.subheader("Train a Model")
        test_size = st.slider("Select Test Size (%):", 10, 50, 20, step=5) / 100
        features = [col for col in data.columns if col not in ['monthly_consumption', 'client_id', 'invoice_date']]
        X = data[features]
        y = data['monthly_consumption']
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = RandomForestRegressor(random_state=42)
            # model = XGBRegressor(random_state=42)  # Uncomment to use XGBoost
            model.fit(X_train, y_train)
            st.session_state.model = model
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model Evaluation Metrics**")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MSE: {mse:.2f}")
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"R²: {r2:.2f}")
            with col2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                ax.set_xlabel("Actual Consumption")
                ax.set_ylabel("Predicted Consumption")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)
            
            # Save and download model
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="model.joblib">Download Model</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("Model trained and ready for download!")
        
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

    # Prediction Section
    with st.expander("🔍 Make Predictions"):
        st.subheader("Predict Monthly Consumption")
        prediction_method = st.radio("Choose Prediction Method:", ["Manual Input", "Select Sample Row"])
        
        if prediction_method == "Manual Input":
            st.write("Enter Feature Values:")
            input_data = {}
            for feature in features:
                input_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)
            input_df = pd.DataFrame([input_data])
            
            if st.button("Predict"):
                if st.session_state.model:
                    try:
                        prediction = st.session_state.model.predict(input_df)
                        st.success(f"Predicted Monthly Consumption: {prediction[0]:.2f}")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                else:
                    st.warning("Please train a model first!")
        
        else:
            sample_row = st.selectbox("Select a sample row:", data.index)
            sample_data = data.loc[[sample_row], features]
            st.write("Selected Row Data:")
            st.dataframe(sample_data)
            
            if st.button("Predict"):
                if st.session_state.model:
                    try:
                        prediction = st.session_state.model.predict(sample_data)
                        st.success(f"Predicted Monthly Consumption: {prediction[0]:.2f}")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                else:
                    st.warning("Please train a model first!")
else:
    st.warning("Please upload a CSV file to begin analysis.")