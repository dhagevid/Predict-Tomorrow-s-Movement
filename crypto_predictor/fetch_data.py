import yfinance as yf
import pandas as pd
import os

def fetch_crypto_data(ticker="BTC-USD", period="5y"):
    """
    Fetches real-time structured historical data for a given ticker from Yahoo Finance.
    """
    print(f"Fetching real-time data for {ticker} from Yahoo Finance...")
    crypto = yf.Ticker(ticker)
    df = crypto.history(period=period)
    
    if df.empty:
        raise ValueError(f"Failed to fetch data for {ticker}. Check internet connection or valid ticker.")
        
    print(f"Successfully fetched {len(df)} rows of data.")
    
    # Save the raw data for records
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{ticker}_raw.csv"
    df.to_csv(file_path)
    print(f"Raw data saved to {file_path}")
    
    return df

if __name__ == "__main__":
    fetch_crypto_data()
