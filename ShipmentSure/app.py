import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="ShipmentSure – Delivery Prediction",
    page_icon="🚚",
    layout="centered"
)

# ---------------------------------------------------------
# Dark + Light Mode CSS
# ---------------------------------------------------------
st.markdown("""
    <style>

    /* ---- Detect Dark Mode ---- */
    @media (prefers-color-scheme: dark) {
        .title-card, .input-card, .output-card {
            background: #1e1f22 !important;
            color: #f5f5f5 !important;
            box-shadow: 0 4px 20px rgba(255,255,255,0.05) !important;
        }

        .predict-btn button {
            background-color: #3b82f6 !important;
            color: white !important;
        }
    }

    /* ---- Light Mode ---- */
    @media (prefers-color-scheme: light) {
        .title-card, .input-card, .output-card {
            background: white !important;
            color: #333 !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
        }

        .predict-btn button {
            background-color: #2563eb !important;
            color: white !important;
        }
    }

    /* ---- Common Styling ---- */
    .title-card {
        padding: 25px 35px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
    }

    .input-card {
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
    }

    .output-card {
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
    }

    .predict-btn button {
        border-radius: 10px !important;
        padding: 10px 25px !important;
        font-size: 18px !important;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Title Card
# ---------------------------------------------------------
st.markdown("""
<div class="title-card">
    <h1 style='margin-bottom:5px;'>🚚 ShipmentSure</h1>
    <h3 style='opacity:0.8;'>Delivery Status Prediction App</h3>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
model = pickle.load(open(r"C:\Users\DELL\Downloads\ShipmentSure\best_model.pkl", "rb"))
train_columns = pickle.load(open(r"C:\Users\DELL\Downloads\ShipmentSure\train_columns.pkl", "rb"))

# ---------------------------------------------------------
# Input Form Card
# ---------------------------------------------------------
with st.container():
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    st.subheader("Enter Shipment Details")

    col1, col2 = st.columns(2)

    with col1:
        discount = st.number_input("Discount Offered (%)", 0, 100)
        prior_purchases = st.number_input("Prior Purchases", 0, 100)
        weight = st.number_input("Weight (grams)", 0, 40000)
        rating = st.number_input("Customer Rating", 1, 5)

    with col2:
        cost = st.number_input("Cost of Product", 0, 50000)
        calls = st.number_input("Customer Care Calls", 0, 10)
        mode = st.selectbox("Mode of Shipment", ["Ship", "Road", "Flight"])
        gender = st.selectbox("Gender", ["M", "F"])
        warehouse = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Prepare Input Data
# ---------------------------------------------------------
input_data = pd.DataFrame(columns=train_columns)
row = {
    "Discount_offered": discount,
    "Prior_purchases": prior_purchases,
    "Weight_in_gms": weight,
    "Cost_of_the_Product": cost,
    "Customer_rating": rating,
    "Customer_care_calls": calls,
    f"Mode_of_Shipment_{mode}": 1,
    f"Gender_{gender}": 1,
    f"Warehouse_block_{warehouse}": 1
}

# Fill missing features
for col in train_columns:
    if col not in row:
        row[col] = 0

input_data.loc[0] = row

ON_TIME_THRESHOLD = 0.25

# ---------------------------------------------------------
# Predict Button
# ---------------------------------------------------------
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
predict_btn = st.button("Predict Delivery Status")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction Output
# ---------------------------------------------------------
if predict_btn:
    prob_on_time = model.predict_proba(input_data)[0][1]

    st.markdown("<div class='output-card'>", unsafe_allow_html=True)

    st.write(f"### 🔍 Probability of ON-TIME delivery: **{prob_on_time:.3f}**")

    if prob_on_time >= ON_TIME_THRESHOLD:
        st.success("📦 The shipment is predicted **ON TIME**")
    else:
        st.error("⏳ The shipment is predicted **LATE**")

    st.markdown("</div>", unsafe_allow_html=True)
