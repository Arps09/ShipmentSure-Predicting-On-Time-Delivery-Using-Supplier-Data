<h1 align="center">🚚 ShipmentSure – Predicting On-Time Delivery Using Supplier Data</h1>
<p align="center">
  <b>A Machine Learning Project by Arpita Mishra</b>
</p>

---

## 🚀 Project Overview

**ShipmentSure** is a machine-learning powered application designed to predict whether a shipment will arrive **on time** or **late**, based on supplier behavior, product characteristics, and logistical features.

The project covers the complete end-to-end ML workflow:

✔ Data Cleaning & Preprocessing  
✔ Exploratory Data Analysis (EDA)  
✔ Handling Class Imbalance  
✔ Feature Engineering  
✔ Model Training & Evaluation  
✔ Saving Model Artifacts  
✔ Web App Deployment using Streamlit  

---

## 📊 Dataset Information

- **Total Records:** 10,999  
- **Features:** 12 (8 numerical + 4 categorical)  
- **Target Variable:** `Reached.on.Time_Y.N`  
  - **0 → On Time**  
  - **1 → Late**

### 📌 Class Distribution
- **Late:** ~60%  
- **On-Time:** ~40%  

---

## 🔍 Exploratory Data Analysis (EDA)

### ✔ Univariate Analysis
- Warehouse Block: Spread across A–F  
- Mode of Shipment: Mostly **Ship**, followed by Flight and Road  
- Product Importance: Mostly **Low**  
- Customer Rating: Uniform from 1–5  
- Discounts: Skewed towards lower values  
- Weight: Long-tail distribution  

### ✔ Bivariate Insights
- Higher **discounts** → More late deliveries  
- More **customer care calls** → More delays  
- Higher **weight** → Later delivery likelihood increases  
- **Flight** shipments have better delivery performance  

### ✔ Correlation Highlights
- Strong positive correlation: **Cost ↔ Weight**  
- Higher discounts associated with **late deliveries**  

### 🔸 Distribution of Target Variable
<p align="center">
  <img src="assets/eda/target_distribution.png" alt="Target Distribution" width="600"/>
</p>

---

### 🔸 Mode of Shipment Distribution
<p align="center">
  <img src="assets/eda/mode_of_shipment.png" alt="Mode of Shipment" width="600"/>
</p>

---

### 🔸 Discount Offered vs Delivery Status
<p align="center">
  <img src="assets/eda/discount_vs_status.png" alt="Discount vs Status" width="600"/>
</p>

---

### 🔸 Correlation Heatmap
<p align="center">
  <img src="assets/eda/correlation_heatmap.png" alt="Correlation Heatmap" width="600"/>
</p>

---

### 🔸 Weight Distribution
<p align="center">
  <img src="assets/eda/weight_distribution.png" alt="Weight Distribution" width="600"/>
</p>

---

## ⚖ Handling Class Imbalance

The target variable was imbalanced:
- **Class 1 (Late): 6563**
- **Class 0 (On Time): 4436**

To solve this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to balance the training data.

---

## 🛠 Data Preprocessing

- Checked for Missing Values → **None found**
- **Label Encoding** → Gender, Product Importance  
- **One-Hot Encoding** → Warehouse Block, Mode of Shipment  
- **Feature Scaling** → StandardScaler  
- **Train–Test Split** → 80:20 (stratified)
- Saved all processed datasets for reuse  

---

## 🤖 Machine Learning Models Used

The following models were trained and evaluated:

- XGBoost Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- Logistic Regression  

### Evaluation Metrics
- Accuracy  
- ROC–AUC  
- Precision & Recall  
- F1-score  
- ROC Curves  

### ⭐ Best Model  
**Random Forest Classifier** performed the best and was selected for deployment.

---

## 💾 Saving Trained Models

All trained ML models were saved as `.pkl` files using `joblib.dump()`.

Deployed models:
- **best_model.pkl**
- **train_columns.pkl**

---

## 🌐 Deployment Using Streamlit

The Random Forest ML model is deployed as an **interactive Streamlit web app**.

### ✔ User Inputs
- Discount Offered  
- Prior Purchases  
- Weight (grams)  
- Mode of Shipment  
- Gender  
- Customer Care Calls  
- Cost of Product  
- Customer Rating  
- Warehouse Block  

### ✔ Model Outputs
- **Prediction:** On Time / Late  
- **Prediction Probability**  
- Visual + Color-coded interpretation  

---

## 🚀 Live Application

🔗 **Live Web App:**  
👉 https://shipmentsure.streamlit.app/

---

## ▶ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
