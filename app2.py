import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load the pre-trained model
model = joblib.load("xgb_model.pkl")  # Replace with your .pkl file path

# Load the dataset dynamically to calculate mean price per cent
data_path = "Updated_Cleaned_Dataset (1).csv"  # Replace with your dataset path
dataset = pd.read_csv(data_path)

# Calculate mean price per cent dynamically
dataset['Mean_Price_Per_Cent'] = dataset.groupby('Standardized_Location_Name')['Price_per_cent'].transform('mean')
location_mean_price_per_cent = dataset.groupby('Standardized_Location_Name')['Price_per_cent'].mean().to_dict()

# PolynomialFeatures setup
numerical_features = ['Build__Area', 'Total_Area', 'Mean_Price_Per_Cent']
poly_feature_names = [
    'Build__Area', 'Total_Area', 'Mean_Price_Per_Cent',
    'Build__Area^2', 'Build__Area Total_Area', 'Build__Area Mean_Price_Per_Cent',
    'Total_Area^2', 'Total_Area Mean_Price_Per_Cent', 'Mean_Price_Per_Cent^2'
]

# Function to predict price
def predict_price(build_area, plot_area_cents, bedrooms, location):
    mean_price_per_cent = location_mean_price_per_cent.get(location, np.mean(list(location_mean_price_per_cent.values())))
    total_area = build_area + (plot_area_cents * 435.6)  # Approximate conversion from cents to sq ft

    input_data = {
        'Build__Area': build_area,
        'Total_Area': total_area,
        'Mean_Price_Per_Cent': mean_price_per_cent,
    }

    # Simulate PolynomialFeatures transformation
    input_poly = [
        input_data['Build__Area'],
        input_data['Total_Area'],
        input_data['Mean_Price_Per_Cent'],
        input_data['Build__Area'] ** 2,
        input_data['Build__Area'] * input_data['Total_Area'],
        input_data['Build__Area'] * input_data['Mean_Price_Per_Cent'],
        input_data['Total_Area'] ** 2,
        input_data['Total_Area'] * input_data['Mean_Price_Per_Cent'],
        input_data['Mean_Price_Per_Cent'] ** 2
    ]

    input_df = pd.DataFrame([input_poly], columns=poly_feature_names)

    # Simulate one-hot encoding for location
    location_dummies = pd.DataFrame([{loc: 1 if loc == location else 0 for loc in location_mean_price_per_cent.keys()}])

    # Combine all inputs
    full_input = pd.concat([input_df, location_dummies], axis=1)
    full_input = full_input.reindex(columns=poly_feature_names + list(location_mean_price_per_cent.keys()), fill_value=0)

    # Predict using the model
    return model.predict(full_input)[0]

# Streamlit App
st.title("Real Estate Price Predictor")
st.write("Predict the price of a plot based on various features.")

# User Inputs
build_area = st.number_input("Enter Build Area (sq ft):", min_value=500, step=50)
plot_area_cents = st.number_input("Enter Plot Area (cents):", min_value=1, step=1)
bedrooms = st.number_input("Enter Bedroom Count:", min_value=1, step=1)
location = st.selectbox("Select Location:", list(location_mean_price_per_cent.keys()))

# Prediction
if st.button("Predict Price"):
    price = predict_price(build_area, plot_area_cents, bedrooms, location)
    st.write(f"### Predicted Price: ₹{price:,.2f}")
