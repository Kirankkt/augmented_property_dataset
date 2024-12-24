import pandas as pd
import numpy as np
import streamlit as st
import joblib
from babel.numbers import format_currency

# Load the pre-trained model and feature names
model = joblib.load("xgb_model.pkl")  # Replace with your .pkl file path
feature_names = joblib.load("xgb_feature_names.pkl")  # Replace with your .pkl file path

# Load the dataset dynamically to calculate mean price per cent
data_path = "Updated_Cleaned_Dataset (1).csv"  # Replace with your dataset path
dataset = pd.read_csv(data_path)

# Calculate mean price per cent dynamically
dataset['Mean_Price_Per_Cent'] = dataset.groupby('Standardized_Location_Name')['Price_per_cent'].transform('mean')
location_mean_price_per_cent = dataset.groupby('Standardized_Location_Name')['Price_per_cent'].mean().to_dict()
location_plot_counts = dataset['Standardized_Location_Name'].value_counts().to_dict()

# PolynomialFeatures setup (columns must match the training pipeline)
numerical_features = ['Build__Area', 'Total_Area', 'Mean_Price_Per_Cent']
poly_feature_names = [
    'Build__Area', 'Total_Area', 'Mean_Price_Per_Cent',
    'Build__Area^2', 'Build__Area Total_Area', 'Build__Area Mean_Price_Per_Cent',
    'Total_Area^2', 'Total_Area Mean_Price_Per_Cent', 'Mean_Price_Per_Cent^2'
]

# Function to format price using Babel
def format_price_indian(price):
    return format_currency(price, 'INR', locale='en_IN')

# Function to predict price
def predict_price(build_area, plot_area_cents, bedrooms, location):
    # Get mean price per cent for the selected location
    mean_price_per_cent = location_mean_price_per_cent.get(
        location, 
        np.mean(list(location_mean_price_per_cent.values()))  # Fallback to global mean
    )
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

    # One-hot encode location and align with training feature names
    location_dummies = pd.DataFrame([{loc: 1 if loc == location else 0 for loc in location_mean_price_per_cent.keys()}])
    full_input = pd.concat([input_df, location_dummies], axis=1)
    full_input = full_input.reindex(columns=feature_names, fill_value=0)

    # Predict using the model
    return model.predict(full_input)[0], mean_price_per_cent

# Streamlit App
st.title("Real Estate Price Predictor")
st.write("Predict the price of a plot based on various features.")

# User Inputs
build_area = st.number_input("Enter Build Area (sq ft):", min_value=500, step=50)
plot_area_cents = st.number_input("Enter Plot Area (cents):", min_value=1, step=1)
bedrooms = st.number_input("Enter Bedroom Count:", min_value=1, step=1)
location = st.selectbox("Select Location:", sorted(location_mean_price_per_cent.keys()))

# Prediction
if st.button("Predict Price"):
    price, mean_price_per_cent = predict_price(build_area, plot_area_cents, bedrooms, location)
    formatted_price = format_price_indian(price)
    plot_count = location_plot_counts.get(location, 0)
    
    st.write(f"### Predicted Price: {formatted_price}")
    format_mean_price=format_price_indian(mean_price_per_cent)
    st.write(f"Mean Price per Cent for '{location}': ₹{format_mean_price}")
    st.write(f"Number of plots available in '{location}': {plot_count}")
