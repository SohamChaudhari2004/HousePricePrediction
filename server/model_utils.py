import joblib
import numpy as np

# Load model & scaler once when the app starts
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/house_price_scaler.pkl")

def predict_house_price(features: list) -> float:
    """Predicts house price based on input features."""
    input_data = np.array([features])  # Convert to 2D array
    scaled_prediction = model.predict(input_data)  # Predict price (scaled)
    original_price = scaler.inverse_transform(scaled_prediction.reshape(-1, 1))  # Inverse transform
    
    return round(float(original_price[0][0]), 2)  # Return as float
