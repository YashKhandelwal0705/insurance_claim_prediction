import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os

# Add the 'src' directory to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils.app_utils import engineer_features, reorder_columns

# --- Page Configuration ---
st.set_page_config(
    page_title="Insurance Claim Severity Predictor",
    page_icon="🛡️",
    layout="wide"
)

# --- Model & Artifact Loading ---
@st.cache_resource
def load_artifacts():
    """Load the trained model, preprocessor, and SHAP explainer."""
    try:
        model = joblib.load('models/best_model.pkl')
        preprocessor = joblib.load('models/tree_preprocessor.pkl')
        shap_explainer = joblib.load('models/shap_explainer.pkl')
        return model, preprocessor, shap_explainer
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure all artifacts are in the 'models' directory.")
        return None, None, None

# --- Business Insights Generation ---
def get_business_insights(df):
    """Generate business insights with icons based on input data."""
    insights = []
    if df['driver_age'].iloc[0] <= 25:
        insights.append("🧑‍🦱 **Young Driver Risk:** Drivers under 25 often have higher claim rates due to less experience.")
    if df['vehicle_make'].iloc[0] == 'High-end':
        insights.append("🚗 **High-End Vehicle:** Luxury vehicles have higher repair costs, leading to more severe claims.")
    if df['accident_type'].iloc[0] == 'Severe':
        insights.append("💥 **Severe Accident:** The accident type is a strong driver of high claim costs.")
    if df['past_claims'].iloc[0] > 1:
        insights.append("📜 **Claim History:** A history of multiple past claims is a significant indicator of future risk.")
    if not insights:
        insights.append("✅ **Low Risk Profile:** This profile does not match common high-risk factors. Review the SHAP plot for detailed drivers.")
    return insights

# --- Main Application ---
st.title("🛡️ Insurance Claim Severity Predictor")
st.markdown("Welcome! This tool predicts the potential severity of an insurance claim based on policy and vehicle details. Please enter the information in the sidebar to get a prediction.")

model, preprocessor, shap_explainer = load_artifacts()

if model is not None:
    # --- Sidebar for User Input ---
    st.sidebar.header('Enter Policy & Vehicle Details')
    driver_age = st.sidebar.slider('Driver Age', 18, 80, 35)
    past_claims = st.sidebar.number_input('Number of Past Claims', min_value=0, max_value=20, value=0)
    vehicle_age = st.sidebar.slider('Vehicle Age (years)', 0, 20, 5)
    vehicle_type = st.sidebar.selectbox('Vehicle Type', ['Sedan', 'SUV', 'Sports', 'Truck'])
    vehicle_make = st.sidebar.selectbox('Vehicle Make', ['Regular', 'High-end'])
    region = st.sidebar.selectbox('Region', ['Suburban', 'Urban', 'Rural'])
    accident_type = st.sidebar.selectbox('Accident Type', ['Moderate', 'Minor', 'Severe'])

    if st.sidebar.button('Predict Claim Severity', type="primary"):
        input_df = pd.DataFrame({
            'driver_age': [driver_age], 'past_claims': [past_claims], 'vehicle_age': [vehicle_age],
            'vehicle_type': [vehicle_type], 'vehicle_make': [vehicle_make], 'region': [region],
            'accident_type': [accident_type]
        })

        # --- Prediction and Explainability Pipeline ---
        engineered_df = engineer_features(input_df.copy())
        ordered_df = reorder_columns(engineered_df)
        prediction_log = model.predict(ordered_df)
        prediction = np.expm1(prediction_log)[0]
        # Get transformed features from the model's preprocessor for SHAP
        input_transformed = model.named_steps['preprocessor'].transform(ordered_df)
        shap_values = shap_explainer(input_transformed)

        # --- Display Dashboard ---
        st.header("Prediction Results")
        tab1, tab2 = st.tabs(["📈 Prediction & Insights", "🧠 Prediction Deep Dive"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Predicted Claim Amount")
                st.metric(label="Severity", value=f"₹{prediction:,.2f}", delta_color="inverse")
                st.caption("This is the estimated cost of the claim based on the provided details.")
            
            with col2:
                st.subheader("Key Risk Factors")
                insights = get_business_insights(input_df)
                for insight in insights:
                    st.markdown(f"- {insight}")

        with tab2:
            st.subheader("Prediction Explainability (SHAP Analysis)")
            st.markdown("The waterfall plot below shows how each feature contributed to pushing the prediction away from the average claim amount. Features in red increased the predicted cost, while those in blue decreased it.")
            fig, ax = plt.subplots()
            shap.waterfall_plot(shap_values[0], max_display=10, show=False)
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("Enter details in the sidebar and click 'Predict Claim Severity' to see the results.")
else:
    st.error("Application cannot start because model artifacts are missing. Please check the logs.")
