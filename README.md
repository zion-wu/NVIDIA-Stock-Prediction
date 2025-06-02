# NVIDIA Stock Price Forecasting with LSTM

This project applies Long Short-Term Memory (LSTM) models to predict the stock price movements of NVIDIA (NVDA) using daily adjusted closing prices over the past five years. The goal is to evaluate the effectiveness of LSTM architectures in forecasting financial time series data, with a focus on hyperparameter tuning to improve predictive performance.

## 📊 Research Question

**In layman's terms**:  
Can we use machine learning, specifically LSTM networks, to predict future stock prices based on past performance? Why does this matter?

**Why it matters**:
- Stock price forecasting is critical for investors and traders to make informed decisions, manage risks, and optimize portfolios.
- Accurate predictions can inform **trading strategies**, **risk management**, and **automated systems** like algorithmic trading bots.
- Financial time series are **sequential** in nature; thus, models like LSTMs that can capture long-term dependencies may offer advantages over traditional methods.

This project demonstrates the application of deep learning to real-world finance, leveraging techniques from research such as LeCun, Bengio, and Hinton (2015).

## 📉 Dataset

- **Source**: [Yahoo Finance](https://finance.yahoo.com/quote/NVDA)
- **Stock**: NVIDIA (NVDA)
- **Time Frame**: 5 years of daily adjusted closing prices
- **Target Variable**: `Close` price

Data preprocessing steps:
- **Scaling**: Applied MinMaxScaler to normalize data between 0 and 1.
- **Sliding Window**: 60-day lookback window to predict the next day's price.
- **Train/Test Split**: 80% training, 20% testing (chronological split to maintain time order).

## 🧰 Methodology

### 1️⃣ Baseline LSTM Model
- Two LSTM layers (50 units each)
- Activation: ReLU
- Dropout: 0.2
- Dense output layer with 1 neuron
- Optimizer: Adam
- Early Stopping (patience=10) based on validation loss
- Trained for up to 50 epochs, batch size=32

### 2️⃣ Hyperparameter-Tuned LSTM Model
- **Hyperparameters Tuned** (via Keras Tuner with Hyperband):
  - LSTM units (50–200)
  - Dropout rate (0.1–0.5)
  - Learning rate
  - Batch size
- Goal: Minimize validation loss and improve generalization.

### Cross-Validation:
Used validation split (10% of training data) for performance monitoring.

## 📊 Evaluation

Metrics:
- **Mean Squared Error (MSE)** on test set
- Training & validation loss curves
- Graphs:
  - True vs. Predicted prices
  - Forecast for next 30 days

### Key Results:

| Model                           | Key Findings                                   |
|--------------------------------|------------------------------------------------|
| **Baseline LSTM**               | Captures overall trend, struggles with volatility |
| **Tuned LSTM**                  | Lower validation loss, improved stability, better generalization |

## 📈 Visualizations

- Training vs. Validation Loss Curves
- True vs. Predicted Price Charts
- 30-Day Future Price Forecasts
