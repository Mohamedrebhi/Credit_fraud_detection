from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import Dict, List
import uvicorn

app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time credit card fraud detection",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Fraud Detection API",
        "endpoints": {
            "/predict": "POST - Predict fraud probability for a transaction",
            "/docs": "GET - API documentation"
        }
    }

class Transaction(BaseModel):
    features: List[float]

class FraudDetectionAPI:
    def __init__(self, model_path: str):
        self.supervised_model = joblib.load(f"{model_path}/supervised_model.joblib")
        self.anomaly_detector = joblib.load(f"{model_path}/anomaly_detector.joblib")

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        # Get predictions from both models
        supervised_prob = self.supervised_model.predict_proba(features.reshape(1, -1))[0][1]
        anomaly_score = self.anomaly_detector.score_samples(features.reshape(1, -1))[0]
        
        # Normalize anomaly score to [0,1] range
        normalized_anomaly = 1 / (1 + np.exp(anomaly_score))
        
        # Combine both predictions with weighted average
        combined_prob = (0.7 * supervised_prob + 0.3 * normalized_anomaly)
        
        return {
            "fraud_probability": float(combined_prob),
            "anomaly_score": float(normalized_anomaly)
        }

detector = FraudDetectionAPI("models")

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    try:
        features = np.array(transaction.features)
        prediction = detector.predict(features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)