import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from typing import List, Any
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="hand gesture prediction")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model_path = MODEL_DIR / "SVC_classifier.pkl"
transformer_path = MODEL_DIR / "transformer.pkl"
label_encoder_path = MODEL_DIR / "target_encoder.pkl"


# Load the trained model and transformer
logger.info(f"Loading model from {model_path} and transformers from {transformer_path}")
if not model_path.exists() or not transformer_path.exists():
    raise FileNotFoundError("Model or transformer file not found")

model = joblib.load(model_path)
transformer = joblib.load(transformer_path)
label_encoder = joblib.load(label_encoder_path)




# Define Pydantic input model
class GestureInput(BaseModel):
    data: List[List[Any]]

columns = [
    f"{axis}{i}" for i in range(1, 22) for axis in ["x", "y", "z"]
]

gesture_to_command = {
    "like": "up",
    "two_up": "down",
    "fist": "left",
    "peace": "right"
}

# Define prediction endpoint
@app.post("/predict")
async def predict(payload: GestureInput):
    logger.info("Received prediction request.")
    try:
        # Convert to DataFrame
   
        df = pd.DataFrame(payload.data, columns=columns)

        # Transform data
        X_transformed = transformer.transform(df)

        # Predict
        prediction_encoded = model.predict(X_transformed)

        # Decode label
        prediction_decoded = label_encoder.inverse_transform(prediction_encoded)

        logger.info(f"Prediction: {prediction_decoded[0]}")

        gesture = prediction_decoded[0]
        command = gesture_to_command.get(gesture, "unknown")

        return JSONResponse({
            "prediction": prediction_decoded[0],
            "command": command,
            "message": "Prediction successful"
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
