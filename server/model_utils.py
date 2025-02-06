import joblib
import pandas as pd
import numpy as np
def load_model(model_path = "models/house_price_model.pkl"):
    return joblib.load(model_path)

def predict_price(data, model) -> float:
    input_data = np.array([
        data.area,
        data.bedrooms,
        data.bathrooms,
        data.stories,
        data.parking,
        data.mainroad_yes,
        data.guestroom_yes,
        data.basement_yes,
        data.hotwaterheating_yes,
        data.airconditioning_yes,
        data.prefarea_yes,
        data.furnishingstatus_semi_furnished,
        data.furnishingstatus_unfurnished
    ]).reshape(1, -1)  # Ensure input is a 2D array
    
    prediction = model.predict(input_data)  # Predict using the model
    prediction_value = float(prediction[0])  # Extract and convert the first value to float
    return prediction_value

