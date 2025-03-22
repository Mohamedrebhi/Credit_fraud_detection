import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
from typing import Dict, Any

class FraudDetectionModel:
    def __init__(self):
        self.supervised_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        self.anomaly_detector = IsolationForest(
            n_estimators=150,
            contamination=0.002,  # Adjusted for credit card fraud rate
            max_samples='auto',
            random_state=42
        )
        
    def train_supervised(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the supervised model"""
        self.supervised_model.fit(X_train, y_train)
        
    def train_unsupervised(self, X_train: np.ndarray) -> None:
        """Train the unsupervised anomaly detector"""
        self.anomaly_detector.fit(X_train)
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.supervised_model.predict(X_test)
        
        return {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred)
        }
    
    def save_models(self, path: str) -> None:
        """Save trained models"""
        joblib.dump(self.supervised_model, f"{path}/supervised_model.joblib")
        joblib.dump(self.anomaly_detector, f"{path}/anomaly_detector.joblib")