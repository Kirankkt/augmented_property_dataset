# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures

# Load the pre-trained model
model = joblib.load("xgb_model.pkl")

# Streamlit app title
st.title("Plot Price Prediction")

# User inputs
st.sidebar.header("Input Features")
build_area = st.sidebar.number_input("Build Area (sqft)", min_value=500, max_value=10000, step=50, value=1500)
plot_area_cents = st.sidebar.number_input("Plot Area (cents)", min_value=1, max_value=100, step=1, value=5)
bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
location = st.sidebar.text_input("Enter Location (e.g., kazhakkoottam)", value="kazhakkoottam")

# Load training data mean price per cent information (calculated during training)
training_data = pd.read_csv("Updated_Cleaned_Dataset (1).csv")
location_mean_price_per_cent = training_data.groupby("Standardized_Location_Name")["Price_per_cent"].mean()

# Calculate Mean Price Per Cent for input location dynamically
mean_price_per_cent = location_mean_price_per_cent.get(location.lower(), location_mean_price_per_cent.mean())

# Prepare input data for prediction
total_area = build_area + (plot_area_cents * 435.6)  # Approximate conversion from cents to sqft

# Constructing features (matching training setup)
input_data = pd.DataFrame([{
    "Build__Area": build_area,
    "Total_Area": total_area,
    "Mean_Price_Per_Cent": mean_price_per_cent,
    "Location": location.lower()
}])

# Generate polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
numerical_features = input_data[["Build__Area", "Total_Area", "Mean_Price_Per_Cent"]]
poly_features = poly.fit_transform(numerical_features)

# Combine polynomial features with location encoding (dummy variables)
location_dummies = pd.get_dummies([location.lower()], drop_first=True)
poly_feature_names = poly.get_feature_names_out(["Build__Area", "Total_Area", "Mean_Price_Per_Cent"])
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
final_input = pd.concat([poly_df, location_dummies], axis=1).reindex(columns=model.feature_names_in_, fill_value=0)

# Make prediction
predicted_price = model.predict(final_input)[0]

# Display result
st.write(f"### Predicted Plot Price: ₹{predicted_price:,.2f}")
