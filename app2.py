# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures

# Load the pre-trained model
model = joblib.load("xgb_model.pkl")

# Load the location mean price per cent map
location_mean_price_per_cent = joblib.load("location_mean_price.pkl")

# Ensure the location_mean_price_per_cent is a dictionary
if isinstance(location_mean_price_per_cent, dict):
    mean_price_default = sum(location_mean_price_per_cent.values()) / len(location_mean_price_per_cent)
else:
    # Convert to dictionary if it's a numpy array or similar
    location_mean_price_per_cent = {str(i): val for i, val in enumerate(location_mean_price_per_cent)}
    mean_price_default = sum(location_mean_price_per_cent.values()) / len(location_mean_price_per_cent)

# Initialize polynomial features (ensure it matches the training configuration)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_feature_names = joblib.load("poly_feature_names.pkl")

# Streamlit app title
st.title("Plot Price Prediction")

# User inputs
st.sidebar.header("Input Features")
build_area = st.sidebar.number_input("Build Area (sqft)", min_value=500, max_value=10000, step=50, value=1500)
plot_area_cents = st.sidebar.number_input("Plot Area (cents)", min_value=1, max_value=100, step=1, value=5)
bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
location = st.sidebar.selectbox(
    "Select Location",
    options=list(location_mean_price_per_cent.keys()),
    index=0
)

# Prepare input data
mean_price_per_cent = location_mean_price_per_cent.get(location, mean_price_default)

input_data = {
    'Build__Area': build_area,
    'Total_Area': build_area + (plot_area_cents * 435.6),  # Approximate conversion from cents to sqft
    'Mean_Price_Per_Cent': mean_price_per_cent,
}
input_poly = poly.transform([list(input_data.values())])
input_df = pd.DataFrame(input_poly, columns=poly_feature_names)

# Ensure alignment of dummy variables
location_dummies = pd.get_dummies(pd.Series(location), drop_first=True)
full_input = pd.concat([input_df, location_dummies], axis=1)
X_train_columns = joblib.load("x_train_columns.pkl")
full_input = full_input.reindex(columns=X_train_columns, fill_value=0)

# Make prediction
predicted_price = model.predict(full_input)[0]

# Display result
st.write(f"### Predicted Plot Price: ₹{predicted_price:,.2f}")
