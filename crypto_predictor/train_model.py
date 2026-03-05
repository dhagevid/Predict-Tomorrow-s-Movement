import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from fetch_data import fetch_crypto_data

def preprocess_data(df):
    """
    Creates technical indicators and features from the raw OHLCV price data.
    Our target: Predict if tomorrow's Close price is > today's Close price.
    """
    data = df.copy()
    
    # Calculate simple moving averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    
    # Calculate daily returns percent
    data['Returns'] = data['Close'].pct_change()
    
    # Volatility
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    
    # Target: 1 if tomorrow's close is strictly greater than today's close, else 0
    # We shift the close price backwards by 1 to get tomorrow's close in the current row
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop NaNs that appear because of moving averages and shifting
    data = data.dropna()
    
    return data

def train_model(ticker="BTC-USD"):
    print(f"Starting Training Pipeline for {ticker}")
    
    # Get latest data
    df = fetch_crypto_data(ticker=ticker, period="max")
    
    # Process the data
    print("Preprocessing data and extracting features...")
    processed_df = preprocess_data(df)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 'Returns', 'Volatility']
    
    X = processed_df[features]
    y = processed_df['Target']
    
    # Chronological sort makes time-series splitting easier (don't shuffle randomly!)
    # Scikit-learn's train_test_split has shuffle=False to respect time sequences.
    print(f"Dataset Size: {len(X)} instances.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Training RandomForest model... This might take a few moments.")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating Model Accuracy on Test (Future) Data...")
    y_pred = clf.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{ticker}_rf_model.pkl"
    joblib.dump(clf, model_path)
    
    # Print the last datestamp used in training
    last_date = processed_df.index[-1]
    print(f"Model saved to {model_path}. Last training data point: {last_date}")

if __name__ == "__main__":
    train_model()
