import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fetch_data import fetch_crypto_data
from train_model import preprocess_data

class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

def create_sequences(data, target, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm(ticker="BTC-USD", seq_length=14, epochs=10):
    print(f"Fetching data for {ticker} to train LSTM...")
    df = fetch_crypto_data(ticker=ticker, period="3y")
    processed_df = preprocess_data(df)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 'Returns', 'Volatility']
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(processed_df[features].values)
    targets = processed_df['Target'].values
    
    # Create temporal sequences
    X, y = create_sequences(scaled_features, targets, seq_length)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Split train/test (80/20) chronologically
    split_idx = int(len(X_tensor) * 0.8)
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]
    
    model = CryptoLSTM(input_size=len(features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training PyTorch LSTM Deep Learning Model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        predictions = (test_outputs >= 0.5).float()
        accuracy = (predictions == y_test).float().mean()
        print(f"LSTM Test Accuracy: {accuracy.item() * 100:.2f}%")
        
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{ticker}_lstm.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'features': features,
        'seq_length': seq_length
    }, model_path)
    print(f"LSTM Model saved to {model_path}")
    return model_path

def predict_lstm_realtime(ticker="BTC-USD"):
    model_path = f"models/{ticker}_lstm.pt"
    if not os.path.exists(model_path):
        return None, "Model not trained yet."
        
    checkpoint = torch.load(model_path)
    features = checkpoint['features']
    seq_length = checkpoint['seq_length']
    scaler = checkpoint['scaler']
    
    # Needs enough historical days for the sequence + moving averages
    df = fetch_crypto_data(ticker=ticker, period="6mo")
    processed_df = preprocess_data(df)
    
    if len(processed_df) < seq_length:
        return None, "Not enough data to create a sequence."
        
    recent_data = processed_df[features].values[-seq_length:]
    scaled_recent = scaler.transform(recent_data)
    
    X_tensor = torch.tensor([scaled_recent], dtype=torch.float32)
    
    model = CryptoLSTM(input_size=len(features))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        output = model(X_tensor)
        prob = output.item()
        
    prediction = 1 if prob >= 0.5 else 0
    return prediction, prob

if __name__ == "__main__":
    train_lstm()
