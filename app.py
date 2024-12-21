import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load and Prepare Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Updated_Cleaned_Dataset (1).csv')
    return df

df = load_data()

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("best_xgb_model.pkl")

model = load_model()

# One-Hot Encode the standardized location names
from sklearn.preprocessing import OneHotEncoder
location_encoder = OneHotEncoder()
location_encoded = location_encoder.fit_transform(df[['Standardized_Location_Name']]).toarray()
location_feature_names = location_encoder.get_feature_names_out(['Standardized_Location_Name'])

# Function for user predictions
def predict_price(build_area, plot_area_cents, bedrooms, location):
    build_to_plot_ratio = build_area / (plot_area_cents * 435.6)
    total_area = build_area + (plot_area_cents * 435.6)

    # Create input array with one-hot encoded location
    location_vector = [0] * len(location_feature_names)
    if f"Standardized_Location_Name_{location}" in location_feature_names:
        location_vector[location_feature_names.tolist().index(f"Standardized_Location_Name_{location}")] = 1

    input_data = np.array([[build_area, plot_area_cents, bedrooms, build_to_plot_ratio, total_area] + location_vector])
    predicted_price = model.predict(input_data)[0]
    return predicted_price

# Streamlit UI
st.title("Real Estate Price Prediction App")

st.sidebar.header("Enter Property Details")

# User inputs
build_area = st.sidebar.number_input("Built-Up Area (sq. ft.)", min_value=500, max_value=10000, step=50)
plot_area_cents = st.sidebar.number_input("Plot Area (cents)", min_value=1.0, max_value=100.0, step=0.1)
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
location = st.sidebar.selectbox("Location", sorted(df['Standardized_Location_Name'].unique()))

# Predict button
if st.sidebar.button("Predict Price"):
    if location not in df['Standardized_Location_Name'].unique():
        st.error("Invalid location entered.")
    else:
        predicted_price = predict_price(build_area, plot_area_cents, bedrooms, location)
        st.success(f"The predicted price for the property is ₹{predicted_price:,.2f}")
