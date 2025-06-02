import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="hand gesture prediction")


BASE_DIR = Path(__file__).resolve()
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
        raise HTTPException(status_code=400, detail=str(e))