from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional
import torch
from train import AirQualityNN
import joblib
import uvicorn
import pandas as pd

# Load the scaler and label encoder
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

app = FastAPI(
    title="Air Quality Classification API",
    description="API for predicting air quality based on a variaty measurements",
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model
model = AirQualityNN(input_dim=len(scaler.mean_), num_classes=len(le.classes_))
model.load_state_dict(torch.load("models/air_quality_model.pt"))
model.eval()


class MeasureData(BaseModel):
    Temperature: Optional[float] = Field(None, description="Temperature measurement")
    Humidity: Optional[float] = Field(None, description="Humidity measurement")
    PM2_5: Optional[float] = Field(
        None, alias="PM2.5", description="PM2.5 concentration"
    )
    PM10: Optional[float] = Field(None, description="PM10 concentration")
    NO2: Optional[float] = Field(None, description="NO2 concentration")
    SO2: Optional[float] = Field(None, description="SO2 concentration")
    CO: Optional[float] = Field(None, description="CO concentration")
    Proximity_to_Industrial_Areas: Optional[float] = Field(
        None, description="Distance to industrial areas"
    )
    Population_Density: Optional[float] = Field(
        None, description="Population density in the area"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "Temperature": 25.0,
                "Humidity": 60.0,
                "PM2.5": 15.0,
                "PM10": 30.0,
                "NO2": 40.0,
                "SO2": 20.0,
                "CO": 1.5,
                "Proximity_to_Industrial_Areas": 2.5,
                "Population_Density": 5000.0,
            }
        }
        validate_by_name = True


@app.post("/predict")
async def predict(data: MeasureData):
    try:
        # Convert input data to dictionary and handle missing values
        feature_names = [
            "Temperature",
            "Humidity",
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "Proximity_to_Industrial_Areas",
            "Population_Density",
        ]

        input_dict = {
            "Temperature": data.Temperature,
            "Humidity": data.Humidity,
            "PM2.5": data.PM2_5,
            "PM10": data.PM10,
            "NO2": data.NO2,
            "SO2": data.SO2,
            "CO": data.CO,
            "Proximity_to_Industrial_Areas": data.Proximity_to_Industrial_Areas,
            "Population_Density": data.Population_Density,
        }

        # Create DataFrame with the correct column order
        df = pd.DataFrame([input_dict])[feature_names]

        # Fill missing values with 0
        df = df.fillna(0)

        # Convert to numpy array
        features = df.values

        # Standardize the input
        features_scaled = scaler.transform(features)

        # Convert to tensor
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            outputs = model(features_tensor)
            pred_probs = torch.softmax(outputs, dim=1)
            pred_label = outputs.argmax(1)

        # Get the predicted class label
        predicted_class = le.inverse_transform(pred_label.numpy())[0]
        prediction_probability = float(pred_probs.max())

        return {
            "predicted_class": predicted_class,
            "confidence": prediction_probability,
            "class_probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(le.classes_, pred_probs[0].numpy())
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Air Quality Classification API",
        "usage": "Send POST request to /predict with pollution measurement features",
    }


@app.get("/aqc")
async def aqc_page(request: Request):
    return templates.TemplateResponse("aqc.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=1111, reload=True)
