import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load and clean data
data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data = pd.get_dummies(data, drop_first=True)
data[['Geography_Germany', 'Geography_Spain']] = data[['Geography_Germany', 'Geography_Spain']].astype(int)

# Split features and target
X = data.drop('Exited', axis=1)
y = data['Exited']

# Handle imbalance
X_res, y_res = SMOTE().fit_resample(X, y)

# Scale features
sc = StandardScaler()
X_res_scaled = sc.fit_transform(X_res)

# Train model
rf = RandomForestClassifier()
rf.fit(X_res_scaled, y_res)

# Save model and scaler
joblib.dump(rf, 'churn_predict_model')
joblib.dump(sc, 'scaler.pkl')

import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('churn_predict_model')
sc = joblib.load('scaler.pkl')

st.title("🏦 Bank Customer Churn Prediction Using ML")
st.markdown("### Enter Customer Details")

# Input fields
credit_score = st.number_input("Credit Score", min_value=0)
age = st.number_input("Age", min_value=0)
tenure = st.number_input("Tenure", min_value=0)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.number_input("Number of Products", min_value=0)
has_cr_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
is_active_member = st.selectbox("Is Active Member?", ["No", "Yes"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
geography = st.selectbox("Geography", ["Germany", "Spain", "France"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Encode categorical variables
Geography_Germany = 1 if geography == "Germany" else 0
Geography_Spain = 1 if geography == "Spain" else 0
Gender = 1 if gender == "Male" else 0
HasCrCard = 1 if has_cr_card == "Yes" else 0
IsActiveMember = 1 if is_active_member == "Yes" else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[credit_score, age, tenure, balance,
                            num_of_products, HasCrCard,
                            IsActiveMember, estimated_salary,
                            Geography_Germany, Geography_Spain, Gender]])
    
    input_scaled = sc.transform(input_data)
    result = model.predict(input_scaled)

    if result[0] == 0:
        st.success("✅ Customer is likely to stay.")
    else:
        st.error("⚠️ Customer is likely to churn.")