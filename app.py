import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from typing import List, Any


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="hand gesture prediction")


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model_path = MODEL_DIR / "SVC_classifier.pkl"
transformer_path = MODEL_DIR / "transformer.pkl"
label_encoder_path = MODEL_DIR / "label_encoder.pkl"


# Load the trained model and transformer
logger.info(f"Loading model from {model_path} and transformers from {transformer_path}")
if not model_path.exists() or not transformer_path.exists():
    raise FileNotFoundError("Model or transformer file not found")

model = joblib.load(model_path)
transformer = joblib.load(transformer_path)
label_encoder = joblib.load(label_encoder_path)




# Define Pydantic input model
class GestureInput(BaseModel):
    columns: List[str]
    data: List[List[Any]]

# Define prediction endpoint
@app.post("/predict")
async def predict(payload: GestureInput):
    logger.info("Received prediction request.")
    try:
        # Convert to DataFrame
        df = pd.DataFrame(payload.data, columns=payload.columns)

        # Transform data
        X_transformed = transformer.transform(df)

        # Predict
        prediction_encoded = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed)

        # Decode label
        prediction_decoded = label_encoder.inverse_transform(prediction_encoded)

        logger.info(f"Prediction: {prediction_decoded[0]}")

        return JSONResponse({
            "prediction": prediction_decoded[0],
            "probabilities": probabilities[0].tolist(),
            "message": "Prediction successful"
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))





"""
# Predict endpoint for raw data
@app.post("/predict")
async def predict(data: dict):
    logger.info(f"Received prediction request with data: {data}")
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        required_cols = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", 
                        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns")

        # Preprocess data using the loaded transformer
        X = df.drop("Exited", axis=1, errors='ignore')
        X_transformed = transformer.transform(X)
        X_transformed_df = pd.DataFrame(X_transformed, columns=transformer.get_feature_names_out())

        # Make prediction
        prediction = model.predict(X_transformed_df)
        probability = model.predict_proba(X_transformed_df)[:, 1][0]
        logger.info(f"Prediction made: {prediction[0]}, Probability: {probability}")


        return JSONResponse({
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "message": "Prediction successful"
        })
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))"""