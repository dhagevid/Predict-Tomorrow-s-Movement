import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import base64
from datetime import datetime
import joblib

# Import project functions
from fetch_data import fetch_crypto_data
from train_model import train_model, preprocess_data
from lstm_model import train_lstm, predict_lstm_realtime

st.set_page_config(page_title="Crypto AI Predictor", page_icon="🚀", layout="wide")

st.title("🤖 Real-Time Crypto AI Predictor")
st.markdown("Use Machine Learning and Deep Learning to forecast real-time crypto prices.")

# Sidebar Controls
st.sidebar.header("Control Panel")
ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")
model_type = st.sidebar.radio("AI Model", ["Random Forest", "LSTM (Deep Learning)"])

if st.sidebar.button("Fetch Latest Data"):
    with st.spinner(f"Fetching live data for {ticker}..."):
        try:
            df = fetch_crypto_data(ticker, period="1y")
            st.session_state['df'] = df
            st.sidebar.success("Data fetched!")
        except Exception as e:
            st.sidebar.error(f"Error fetching data: {e}")

if st.sidebar.button("Train Selected Model"):
    with st.spinner(f"Training {model_type} for {ticker}..."):
        try:
            if model_type == "Random Forest":
                train_model(ticker)
            else:
                train_lstm(ticker, epochs=15)
            st.sidebar.success(f"Model {model_type} trained on {ticker}!")
        except Exception as e:
            st.sidebar.error(f"Error during training: {e}")

# Check if we have data in session
if 'df' not in st.session_state:
    st.info("👈 Please click 'Fetch Latest Data' in the sidebar to start.")
else:
    df = st.session_state['df']
    
    # 1. Plot Recent Data
    st.subheader(f"📈 30-Day Snapshot for {ticker}")
    recent_df = df.tail(30)
    
    fig = go.Figure(data=[go.Candlestick(
        x=recent_df.index,
        open=recent_df['Open'],
        high=recent_df['High'],
        low=recent_df['Low'],
        close=recent_df['Close'],
        name="Candlestick"
    )])
    
    # Add Moving average overlay for visual
    recent_df['SMA_10'] = recent_df['Close'].rolling(window=10).mean()
    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['SMA_10'], mode='lines', name='10-Day SMA', line=dict(color='orange')))
    
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Key Metrics
    last_row = df.iloc[-1]
    last_close = float(last_row['Close'])
    prev_close = float(df.iloc[-2]['Close'])
    change_pct = ((last_close - prev_close) / prev_close) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Live Price", f"${last_close:,.2f}", f"{change_pct:.2f}%")
    col2.metric("24h High", f"${float(last_row['High']):,.2f}")
    col3.metric("24h Low", f"${float(last_row['Low']):,.2f}")
    
    st.divider()
    
    # 3. AI Prediction Button
    st.subheader(f"🔮 AI Oracle: {model_type}")
    
    if st.button("🔮 Predict Tomorrow's Price Movement", use_container_width=True):
        with st.spinner("Analyzing real-time features..."):
            pred = None
            prob = None
            
            try:
                if model_type == "Random Forest":
                    import os
                    model_path = f"models/{ticker}_rf_model.pkl"
                    if not os.path.exists(model_path):
                        st.error(f"⚠️ Random Forest model not found for {ticker}. Please train it first.")
                    else:
                        model = joblib.load(model_path)
                        processed_df = preprocess_data(df)
                        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 'Returns', 'Volatility']
                        last_feat = pd.DataFrame([processed_df.iloc[-1][features]])
                        
                        pred = model.predict(last_feat)[0]
                        probabilities = model.predict_proba(last_feat)[0]
                        prob = probabilities[1] if pred == 1 else probabilities[0]
                        
                else: # LSTM
                    pred, prob_val = predict_lstm_realtime(ticker)
                    if pred is None:
                        st.error(f"⚠️ {prob_val} - Please train LSTM first.")
                    else:
                        prob = prob_val if pred == 1 else 1-prob_val
                        
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                
            if pred is not None:
                st.markdown("### **Prediction Results:**")
                if pred == 1:
                    st.success(f"**Action:** 📈 Likely to go **UP**. (Confidence: {prob*100:.1f}%)")
                else:
                    st.error(f"**Action:** 📉 Likely to go **DOWN**. (Confidence: {prob*100:.1f}%)")
                    
                st.progress(float(prob))
                st.caption("AI Confidence Level indicator")
