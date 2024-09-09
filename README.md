
# [Stock Sentiment Prediction App](https://tcmtest.pythonanywhere.com/)

## Overview

This Flask web application predicts stock movements based on technical indicators and sentiment analysis of news headlines and summaries. It uses pre-trained models to analyze the sentiment and predict whether the stock price will go up or down.


## Features

- **Technical Analysis**: Uses historical stock data and technical indicators like RSI, MACD, and moving averages.
- **Sentiment Analysis**: Uses VADER sentiment analysis for headlines and summaries.
- **Prediction Results**: Displays whether the stock is predicted to go up or down based on technical and sentiment models.
- **Responsive Design**: Simple and clean web interface using HTML and CSS.

## Usage

1. **Clone the repository**:
   Clone the app's GitHub repository to your local machine:

   ```bash
   git clone https://github.com/parshverma/StockSentimentalAnalysisApp.git
   
2. **Install dependencies**:
   Make sure you have Python and the required packages installed. You can install the necessary dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   After installing the dependencies, run the Flask application:

   ```bash
   flask run
   ```

3. **Access the app**:
   Open your web browser and go to `https://tcmtest.pythonanywhere.com/` to access the app.
   Click on Predict button to see the stock sentiment (technical & news based) absed on the data provided.

## Project Details

### Technologies Used

- **Flask**: Web framework for Python.
- **VADER**: Sentiment analysis tool from the NLTK package.
- **Joblib**: Used to load the pre-trained machine learning models.
- **Yahoo Finance**: For fetching historical stock data.

