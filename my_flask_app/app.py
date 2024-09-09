from flask import Flask, render_template, request
import yfinance as yf
import joblib
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
# from my_flask_app.model_code import predict_stock_movement  # Import your model function
from model_code import predict_stock_movement

app = Flask(__name__)

sid = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        headline = request.form['headline']
        summary = request.form['summary']
        date = request.form['date']
        close_price = request.form['close_price']

        # Use the predict_stock_movement function from your model code
        movement_technical, movement_sentiment, explanation = predict_stock_movement(
            date, headline, summary, float(close_price), ticker
        )
        
        return render_template('index.html', 
                               technical=movement_technical, 
                               sentiment=movement_sentiment, 
                               explanation=explanation)
    return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
