import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

@st.cache_resource
def load_and_train_model():
    # Load the dataset
    df = pd.read_csv('crop_yield_dataset.csv')
    
    # Define features and target variable
    X = df[['Temperature (C)', 'Rainfall (mm)', 'Humidity (%)', 'Sunlight (hours)', 
            'Soil pH', 'Soil Nitrogen (%)', 'Soil Phosphorus (ppm)', 
            'Soil Potassium (ppm)', 'Altitude (m)', 'Wind Speed (m/s)']]
    y = df['Crop Yield (tons/ha)']
    
    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the Random Forest Regressor
    rf = RandomForestRegressor()
    rf.fit(X_scaled, y)
    
    return rf, scaler, df

rf, scaler, df = load_and_train_model()

# Crop yield prediction function
def crop_yield_prediction(crop, temperature, rainfall, humidity, sunlight, ph, nitrogen, phosphorus, potassium, altitude, wind_speed):
    input_data = scaler.transform([[temperature, rainfall, humidity, sunlight, ph, nitrogen, phosphorus, potassium, altitude, wind_speed]])
    prediction = rf.predict(input_data)[0]
    return f"The predicted crop yield for {crop} under these conditions is {prediction:.2f} tons/ha"

# Streamlit UI
st.title("CROP YEILD PREDICTION")

# Dropdown for crop selection
crop_list = df['Crop'].unique().tolist()
crop_input = st.selectbox("Select Crop Type", crop_list)

# Input fields for user data
temperature_input = st.number_input("Temperature (°C)")
rainfall_input = st.number_input("Rainfall (mm)")
humidity_input = st.number_input("Relative Humidity (%)")
sunlight_input = st.number_input("Sunlight (hours)")
ph_input = st.number_input("Soil pH Value")
nitrogen_input = st.number_input("Soil Nitrogen (%)")
phosphorus_input = st.number_input("Soil Phosphorus (ppm)")
potassium_input = st.number_input("Soil Potassium (ppm)")
altitude_input = st.number_input("Altitude (m)")
wind_speed_input = st.number_input("Wind Speed (m/s)")

# Predict button
if st.button("Predict Crop Yield"):
    prediction = crop_yield_prediction(crop_input, temperature_input, rainfall_input, humidity_input, sunlight_input, ph_input, nitrogen_input, phosphorus_input, potassium_input, altitude_input, wind_speed_input)
    st.write(prediction)
