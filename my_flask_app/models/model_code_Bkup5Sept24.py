import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import os



# # Download the VADER lexicon for sentiment analysis
# nltk.download('vader_lexicon')

# for sentiment analysis using VADER
sid = SentimentIntensityAnalyzer()

# save and load models
# TECH_MODEL_PATH = 'rf_model.pkl'
# SENTIMENT_MODEL_PATH = 'sentiment_rf_model.pkl'
# SCALER_PATH = 'technical_scaler.pkl'
# Define file paths for models and scaler
TECH_MODEL_PATH = '/models/rf_model.pkl'
SENTIMENT_MODEL_PATH = '/models/sentiment_rf_model.pkl'
SCALER_PATH = '/models/technical_scaler.pkl'

print(f"TECH_MODEL_PATH: {TECH_MODEL_PATH}")
print(f"SCALER_PATH: {SCALER_PATH}")



# Fetch historical stock data
def fetch_historical_data(ticker, end_date, days):
    start_date = end_date - timedelta(days=days * 2)
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.asfreq('B').ffill()
    return stock_data.tail(days)

# Calculate RSI
def calculate_rsi(prices, window_length=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate MACD
def calculate_macd(prices, short_span=12, long_span=26):
    short_ema = prices.ewm(span=short_span, adjust=False).mean()
    long_ema = prices.ewm(span=long_span, adjust=False).mean()
    return short_ema - long_ema

# Prepare the dataset (technical features)
# RSI (Relative Strength Index) Calculation - helps identify overbought or oversold conditions.
# MACD (Moving Average Convergence Divergence) Calculation - identifies momentum changes.
# Moving Averages (SMA20 and SMA50)
# SMA20- Simple Moving Average over the last 20 days.
# SMA50: Simple Moving Average over the last 50 days.
# Close_Lag1: The closing price from the previous day.
# Close_Lag2: The closing price from two days ago.

def prepare_technical_dataset(data):
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data['Price_Up'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)

    # Scale only the numerical features
    scaler = MinMaxScaler(feature_range=(0, 1))
    numerical_features = data[['Close', 'Close_Lag1', 'Close_Lag2', 'RSI', 'MACD', 'SMA20', 'SMA50']]
    scaled_numerical_features = scaler.fit_transform(numerical_features)

    target = data['Price_Up'].values
    return scaled_numerical_features, target, scaler

# Train sentiment data separately based on sentiment values
def prepare_sentiment_data(headline_sentiment, summary_sentiment):
    # Label sentiment as 1 (positive), 0 (neutral/negative)
    sentiment_label = 1 if headline_sentiment + summary_sentiment > 0 else 0
    sentiment_features = np.array([[headline_sentiment, summary_sentiment]])  # shape (1, 2)

    return sentiment_features, sentiment_label

# Function to predict stock movement and explain the prediction
def predict_stock_movement(date, headline, summary, close_price, ticker='TSLA'):
    headline_sentiment = sid.polarity_scores(headline)['compound']
    summary_sentiment = sid.polarity_scores(summary)['compound']

    # Weight the sentiment influence
    sentiment_weight = 10
    combined_headline_sentiment = headline_sentiment * sentiment_weight
    combined_summary_sentiment = summary_sentiment * sentiment_weight

    end_date = datetime.strptime(date, '%Y-%m-%d').date()
    for days in range(50, 150, 20):
        data = fetch_historical_data(ticker, end_date, days)
        try:
            # Prepare technical dataset
            scaled_technical_features, target, scaler = prepare_technical_dataset(data)
            if len(target) >= 10:
                break
        except ValueError:
            continue


    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(scaled_technical_features, target)


        # Load technical model and scaler, or raise error if not found
    if os.path.exists(TECH_MODEL_PATH) and os.path.exists(SCALER_PATH):
        rf_model = joblib.load(TECH_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        raise FileNotFoundError(f"Model or Scaler not found. Make sure {TECH_MODEL_PATH} and {SCALER_PATH} exist.")


    # # Train or load technical model
    # if os.path.exists(TECH_MODEL_PATH) and os.path.exists(SCALER_PATH):
    #     rf_model = joblib.load(TECH_MODEL_PATH)
    #     scaler = joblib.load(SCALER_PATH)
    # else:
    #     X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    #     rf_model = RandomForestClassifier(random_state=42)
    #     rf_model.fit(X_train, y_train)
    #     joblib.dump(rf_model, TECH_MODEL_PATH)
    #     joblib.dump(scaler, SCALER_PATH)



    # Technical prediction
    input_technical_features = np.array([close_price, data['Close'].iloc[-1], data['Close'].iloc[-2],
                                         calculate_rsi(data['Close']).iloc[-1],
                                         calculate_macd(data['Close']).iloc[-1],
                                         data['SMA20'].iloc[-1], data['SMA50'].iloc[-1]]).reshape(1, -1)
    scaled_input_technical_features = scaler.transform(input_technical_features)
    technical_prediction = rf_model.predict(scaled_input_technical_features)
    movement_technical = "Up" if technical_prediction[0] == 1 else "Down"

    # # Train or load sentiment model
    # sentiment_features, sentiment_label = prepare_sentiment_data(combined_headline_sentiment, combined_summary_sentiment)
    # if os.path.exists(SENTIMENT_MODEL_PATH):
    #     rf_sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    # else:
    #     rf_sentiment_model = RandomForestClassifier(random_state=42)
    #     rf_sentiment_model.fit(sentiment_features, [sentiment_label])  # Train with direct sentiment label
    #     joblib.dump(rf_sentiment_model, SENTIMENT_MODEL_PATH)

    # Load sentiment model or raise error if not found
    sentiment_features, sentiment_label = prepare_sentiment_data(combined_headline_sentiment, combined_summary_sentiment)
    if os.path.exists(SENTIMENT_MODEL_PATH):
        rf_sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    else:
        raise FileNotFoundError(f"Sentiment model not found. Make sure {SENTIMENT_MODEL_PATH} exists.")


    # Sentiment prediction
    sentiment_prediction = rf_sentiment_model.predict(sentiment_features)
    movement_sentiment = "Up" if sentiment_prediction[0] == 1 else "Down"


    print(f"Technical Prediction for {end_date + timedelta(days=1)}: {movement_technical}")
    print(f"Sentiment Prediction for {end_date + timedelta(days=1)}: {movement_sentiment}")

    # Feature importance explanation (for technical prediction)
    feature_importances = rf_model.feature_importances_
    important_features = np.argsort(feature_importances)[-7:][::-1]
    feature_names = ['Close Price', 'Previous Day Close', 'Two Days Ago Close',
                     'RSI', 'MACD', 'SMA20', 'SMA50']

    explanation = f"The model predicts the stock will go {movement_technical} based on technical parameters due to the following key factors:\n"
    for i in important_features:
        explanation += f"- {feature_names[i]} had a significant value of {scaled_input_technical_features[0][i]:.2f}, which contributed to this prediction.\n"

    explanation += f"\nThe model predicts the stock will go {movement_sentiment} based on news sentiment with a headline sentiment of {combined_headline_sentiment:.2f} and summary sentiment of {combined_summary_sentiment:.2f}.\n"

    print(explanation)
    return movement_technical, movement_sentiment, explanation

# testHeadline = "Tesla Improves Maps and Navigation With Update 2024.32"
# testSummary = "Tesla recently released update 2024.32 to employees, and it comes with several new features that we’re sure everyone will be happy to see in their vehicles in the future.Keep in mind that some features may be region-specific such as there construction on your route update. There may also be additional features available for other models or regions. The Cybertruck is expected to receive AutoPark soon, which may also be included in this update."
# predict_stock_movement('2024-09-2', testHeadline, testSummary, 214.11)

# print(movement_technical)
# print(movement_sentiment)
# print(explanation)