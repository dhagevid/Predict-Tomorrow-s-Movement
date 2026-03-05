import sys
import joblib
import pandas as pd
from fetch_data import fetch_crypto_data

def predict_tomorrow(ticker="BTC-USD"):
    print(f"Loading pre-trained model for {ticker}...")
    model_path = f"models/{ticker}_rf_model.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Please run train_model.py first.")
        sys.exit(1)
        
    print(f"Fetching latest live data from exchange...")
    # Get recent data. We need at least the last 30 days to calculate a 30-day moving average
    df = fetch_crypto_data(ticker, period="60d")
    
    # Calculate features in real-time
    data = df.copy()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    
    # We drop NAs that result from rolling averages
    data = data.dropna()
    
    if data.empty:
        print("Not enough data to calculate features. Try a bigger period.")
        sys.exit(1)
        
    # The last row represents today (the current moment, often partially completed if the market isn't closed)
    last_row = data.iloc[-1]
    last_date = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\nCurrent Data Snapshot: [{last_date}]")
    print(f"Close Price: ${last_row['Close']:.2f}")
    
    # Prepare features for prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 'Returns', 'Volatility']
    
    # Convert latest row to a dataframe of 1 row so the model can predict it
    X_pred = pd.DataFrame([last_row[features]])
    
    prediction = model.predict(X_pred)[0]
    probabilities = model.predict_proba(X_pred)[0]
    
    print("-" * 50)
    if prediction == 1:
        print(f"🤖 AI Prediction: UP ⬆️ (Price is expected to rise by tomorrow's close)")
        print(f"Confidence: {probabilities[1]*100:.1f}%")
    else:
        print(f"🤖 AI Prediction: DOWN ⬇️ (Price is expected to fall by tomorrow's close)")
        print(f"Confidence: {probabilities[0]*100:.1f}%")
    print("-" * 50)

if __name__ == "__main__":
    predict_tomorrow()
