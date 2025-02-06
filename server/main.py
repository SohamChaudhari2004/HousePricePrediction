from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from model_utils import load_model, predict_price

# Define Pydantic model for input data
class HouseData(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    mainroad_yes: int
    guestroom_yes: int
    basement_yes: int
    hotwaterheating_yes: int
    airconditioning_yes: int
    prefarea_yes: int
    furnishingstatus_semi_furnished: int
    furnishingstatus_unfurnished: int

# Define Pydantic model for output response
class PredictionResponse(BaseModel):
    prediction: str  # String to return a formatted price

# Initialize FastAPI app
app = FastAPI()

# Load the model once at startup
model = load_model()

# Prediction function
# def predict_price(data: HouseData, model) -> float:
    

@app.get('/')
def check_app():
    return {"message": "Welcome to the house price prediction model!"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: HouseData):
    try:
        prediction = predict_price(model,data)  # Get raw prediction value
        formatted_prediction = "${:}".format(prediction)  # Format as currency
        return PredictionResponse(prediction=formatted_prediction)  # Return formatted price
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")
