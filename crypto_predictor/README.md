# Real-Time Crypto Price Predictor (AI/ML)

Welcome to your Real-Time Cryptocurrency Prediction AI! 
This project pulls **live, realistic data** from Yahoo Finance (`yfinance`) and uses a **Machine Learning Model** (Random Forest) from `scikit-learn` to predict if the price of an asset (like Bitcoin `BTC-USD`) will close higher or lower tomorrow.

## 🛠 Features
- **Real-Time Data Streaming**: Pulls the very latest ticker data.
- **Feature Engineering**: Automatically calculates Technical Indicators (Moving Averages, Volatility).
- **Time-Series ML Pipeline**: Uses historical sequences to train a Random Forest classifier.
- **Live Predictions**: Predicts tomorrow's market movement with probability estimates.

## 🚀 How to Run It

### Step 1: Install Requirements
Open a terminal in this folder and install the required data science libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Interactive Web Dashboard ✨
The best way to interact with the models is through the web dashboard!
It will open an interactive page where you can fetch the latest data, train the models (Random Forest and Deep Learning LSTM), and make live predictions.
```bash
streamlit run app.py
```

### (Optional) Step 3: Command Line Usage
If you prefer the terminal to the dashboard, you can run the scripts manually:
- **Train Random Forest**: `python train_model.py`
- **Train PyTorch LSTM**: `python lstm_model.py`
- **Predict via Terminal**: `python predict_realtime.py`

---
*Note: This is an educational AI project and does not constitute financial advice!*
