import pandas as pd
from preprocessing import DataPreprocessor
from train_model import FraudDetectionModel
import os

def train_models():
    # Load both datasets
    df1 = pd.read_csv('c:/Users/gigabyte/fraud_detection/data/creditcard.csv')
    df2 = pd.read_csv('c:/Users/gigabyte/fraud_detection/data/credit_card_transactions.csv')
    
    print(f"First dataset size: {df1.shape[0]} transactions")
    print(f"Second dataset size: {df2.shape[0]} transactions")
    
    # Print column names to inspect
    print("\nDataset 1 columns:", df1.columns.tolist())
    print("Dataset 2 columns:", df2.columns.tolist())
    
    # Prepare second dataset to match first dataset structure
    df2_prepared = pd.DataFrame()
    df2_prepared['Time'] = pd.to_datetime(df2['trans_date_trans_time']).map(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    df2_prepared['Amount'] = df2['amt']
    df2_prepared['Class'] = df2['is_fraud'].astype(int)
    
    # Generate meaningful V1-V28 features for second dataset
    df2_prepared['V1'] = (df2['amt'] - df2['amt'].mean()) / df2['amt'].std()  # Amount standardized
    df2_prepared['V2'] = pd.to_datetime(df2['trans_date_trans_time']).dt.hour / 24  # Hour normalized
    df2_prepared['V3'] = df2['category'].astype('category').cat.codes / df2['category'].nunique()  # Category encoded
    df2_prepared['V4'] = df2['merchant'].astype('category').cat.codes / df2['merchant'].nunique()  # Merchant encoded
    df2_prepared['V5'] = (df2['lat'] - df2['lat'].mean()) / df2['lat'].std()  # Latitude standardized
    df2_prepared['V6'] = (df2['long'] - df2['long'].mean()) / df2['long'].std()  # Longitude standardized
    df2_prepared['V7'] = df2.groupby('merchant')['amt'].transform('mean') / df2['amt'].max()  # Merchant avg transaction
    df2_prepared['V8'] = df2.groupby('category')['amt'].transform('mean') / df2['amt'].max()  # Category avg transaction
    
    # Generate remaining V features using combinations of existing features
    for i in range(9, 29):
        if i % 2 == 0:
            df2_prepared[f'V{i}'] = df2_prepared['V1'] * df2_prepared[f'V{i-1}']
        else:
            df2_prepared[f'V{i}'] = df2_prepared['V2'] + df2_prepared[f'V{i-2}']
    
    # Normalize all V features to match the scale of original dataset
    for i in range(1, 29):
        df2_prepared[f'V{i}'] = (df2_prepared[f'V{i}'] - df2_prepared[f'V{i}'].mean()) / df2_prepared[f'V{i}'].std()
    
    # Drop rows with NaN values
    df1 = df1.dropna()
    df2_prepared = df2_prepared.dropna()
    
    # Drop rows with NaN values in the target column
    df1 = df1.dropna(subset=['Class'])
    df2_prepared = df2_prepared.dropna(subset=['Class'])
    
    # Combine datasets
    combined_df = pd.concat([df1, df2_prepared], ignore_index=True)
    print(f"\nCombined dataset size: {combined_df.shape[0]} transactions")
    
    # Verify class distribution
    fraud_count = combined_df['Class'].sum()
    print(f"\nFraud transactions: {fraud_count} ({fraud_count/len(combined_df):.2%})")
    
    # Initialize preprocessor and model
    preprocessor = DataPreprocessor()
    model = FraudDetectionModel()
    
    # Prepare the combined data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(combined_df)
    
    # Train both models
    print("\nTraining supervised model...")
    model.train_supervised(X_train, y_train)
    
    print("Training anomaly detector...")
    model.train_unsupervised(X_train)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the models
    if not os.path.exists('c:/Users/gigabyte/fraud_detection/models'):
        os.makedirs('c:/Users/gigabyte/fraud_detection/models')
    model.save_models('c:/Users/gigabyte/fraud_detection/models')
    print("\nModels saved successfully!")

if __name__ == "__main__":
    train_models()