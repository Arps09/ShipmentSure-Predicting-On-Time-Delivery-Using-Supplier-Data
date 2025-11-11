import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

# ===============================
# 🚚 Shipment Prediction App
# ===============================
st.set_page_config(
    page_title="Shipment Prediction App",
    page_icon="🚚",
    layout="wide"
)

# ===============================
# Load Trained Model
# ===============================
@st.cache_resource
def load_model():
    model_path = os.path.join('saved_models', 'random_forest_model.pkl')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    model = joblib.load(model_path)
    return model

model = load_model()
if model is None:
    st.stop()

# ===============================
# Main App
# ===============================
st.title("🚚 Shipment Prediction System")
st.markdown("Predict whether a shipment will be **On Time** or **Late** using Machine Learning.")

col1, col2 = st.columns(2)

# -------------------------------
# Input Features
# -------------------------------
with col1:
    st.header("📊 Enter Shipment Details")

    mode_of_shipment = st.selectbox("Mode of Shipment", ["Flight", "Ship", "Road"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=100, value=10)
    prior_purchases = st.number_input("Prior Purchases", min_value=0, max_value=20, value=3)
    weight_in_gms = st.number_input("Weight (in grams)", min_value=100, max_value=8000, value=2500)
    customer_care_calls = st.number_input("Customer Care Calls", min_value=0, max_value=10, value=3)
    cost_of_product = st.number_input("Cost of the Product", min_value=10, max_value=10000, value=1000)
    customer_rating = st.slider("Customer Rating", min_value=1, max_value=5, value=3)

# -------------------------------
# Prediction Section
# -------------------------------
with col2:
    st.header("🎯 Prediction Result")

    if st.button("Predict Shipment Status"):
        # Prepare user input
        input_dict = {
            'Discount_offered': discount_offered,
            'Prior_purchases': prior_purchases,
            'Weight_in_gms': weight_in_gms,
            'Mode_of_Shipment_Ship': 1 if mode_of_shipment == "Ship" else 0,
            'Mode_of_Shipment_Road': 1 if mode_of_shipment == "Road" else 0,
            'Mode_of_Shipment_Flight': 1 if mode_of_shipment == "Flight" else 0,
            'Gender_M': 1 if gender == "Male" else 0,
            'Customer_care_calls': customer_care_calls,
            'Cost_of_the_Product': cost_of_product,
            'Customer_rating': customer_rating
        }

        input_df = pd.DataFrame([input_dict])

        # ✅ Align with model's expected features
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Display results
        status = "🟢 On Time" if prediction == 0 else "🔴 Late Delivery"
        st.success(f"**Prediction:** {status}")
        st.info(f"Prediction Probabilities: {prediction_proba}")

        # Chart display
        prob_df = pd.DataFrame({
            'Class': ['On Time', 'Late Delivery'],
            'Probability': prediction_proba
        })
        st.bar_chart(prob_df.set_index('Class'))

# -------------------------------
# Model Info
# -------------------------------
st.header("ℹ️ Model Information")
st.write("Model Type: Random Forest Classifier")
if hasattr(model, 'n_estimators'):
    st.write(f"Number of Trees: {model.n_estimators}")
