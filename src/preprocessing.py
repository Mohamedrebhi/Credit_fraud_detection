import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def set_feature_columns(self, df):
        """Set feature columns excluding the target variable"""
        self.feature_columns = [col for col in df.columns if col != 'Class']
        return self.feature_columns

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features using StandardScaler"""
        if self.feature_columns is None:
            self.set_feature_columns(df)
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for fraud detection"""
        # Time-based features
        df['hour'] = df['Time'] % 24
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Amount-based features
        df['amount_percentile'] = df['Amount'].rank(pct=True)
        df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        
        # Transaction frequency features
        df['transaction_freq'] = df.groupby('hour')['Time'].transform('count')
        df['amount_to_freq_ratio'] = df['Amount'] / (df['transaction_freq'] + 1)
        
        return df

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        if self.feature_columns is None:
            self.set_feature_columns(df)
        X = df[self.feature_columns].values
        y = df['Class'].values
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)