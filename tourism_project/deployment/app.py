"""Streamlit front end for the Wellness Tourism Package purchase prediction model.

Loads the model that the GitHub Actions pipeline commits to tourism_project/deployment/,
collects customer details, puts them into a DataFrame and shows the prediction.
"""
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
THRESHOLD = 0.45  # same threshold used during evaluation

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Visit with Us - Wellness Tourism Package Predictor")
st.write("Enter the customer details below to estimate whether they are likely to buy the "
         "Wellness Tourism Package before the sales team contacts them.")

# ---- Customer details ----
st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", min_value=0.0, value=22000.0, step=500.0)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
with col2:
    passport = st.selectbox("Has a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
    trips = st.number_input("Average trips per year", min_value=0, max_value=30, value=3)
    persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
    children = st.number_input("Children (below 5) visiting", min_value=0, max_value=5, value=1)
    property_star = st.selectbox("Preferred property star", [3, 4, 5])

# ---- Interaction details ----
st.subheader("Sales interaction")
col3, col4 = st.columns(2)
with col3:
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=200, value=15)
with col4:
    followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=3)
    satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

# ---- Collect the inputs into a DataFrame with the same columns used in training ----
input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": contact, "CityTier": city_tier, "DurationOfPitch": duration,
    "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups, "ProductPitched": product, "PreferredPropertyStar": property_star,
    "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction, "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children, "Designation": designation, "MonthlyIncome": income,
}])

with st.expander("Input data sent to the model"):
    st.dataframe(input_df)

if st.button("Predict"):
    proba = model.predict_proba(input_df)[0, 1]
    if proba >= THRESHOLD:
        st.success(f"Likely to purchase the Wellness Tourism Package (probability {proba:.1%}). Prioritise this customer.")
    else:
        st.warning(f"Unlikely to purchase (probability {proba:.1%}). Lower priority for this campaign.")
