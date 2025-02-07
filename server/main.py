from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import predict_house_price

app = FastAPI()

# Define input schema
class HouseFeatures(BaseModel):
    area: float
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

@app.post("/predict")
def predict_price(features: HouseFeatures):
    """API endpoint to predict house price"""
    try:
        # Convert Pydantic model to list of feature values
        feature_list = [
            features.area, features.bedrooms, features.bathrooms,
            features.stories, features.parking, features.mainroad_yes,
            features.guestroom_yes, features.basement_yes, 
            features.hotwaterheating_yes, features.airconditioning_yes,
            features.prefarea_yes, features.furnishingstatus_semi_furnished,
            features.furnishingstatus_unfurnished
        ]
        
        predicted_price = predict_house_price(feature_list)
        return {"predicted_price": predicted_price}

    except Exception as e:
        return {"error": str(e)}

