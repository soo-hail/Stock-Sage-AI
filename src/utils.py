import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import requests
import warnings
import nltk
import os

from nltk.sentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon')


from dotenv import load_dotenv
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# FUNCTION TO LOAD DATA.
@st.cache_data(ttl=3600) # CACHE DATA FOR ONE HOUR.
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True) # MAKE DATE A COLUMN
    
    print(data)

    # FLATTEN MULTIINDEX COLUMNS IF THEY EXIST.
    if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]

    # CALCULATE SIMPLE MOVING AVERAGE
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # CALCULATE EXPOTENTIAL MOVING-AVERAGE
    ema_12 = calculate_ema(data['Close'], 12)
    ema_26 = calculate_ema(data['Close'], 26)
    
    data['EMA12'] = ema_12
    data['EMA26'] = ema_26

    # CALCULATE RSI
    data['RSI'] = calculate_rsi(data['Close'])
    
    return data

# FUNCTION TO CALCULATE RSI.
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# FUNCTION TO CALCULATE EMA.
def calculate_ema(data, period):
    # NOTE: 
    # IN SIMPLE MOVING AVERAGE, ALL WEIGHTS(PRICES - OLD AND NEW) ARE GIVEN EQUAL PRIORITY - SLOW TO SHOW WEIGHT CHANGES. 
    # GIVES MORE PRIORITY TO RECENT PRICES - RECENT PRICES GET MORE PRIORITY(RECENT PRICE MATTERS MORE) - QUICK REFLECTION TO CURRENT WEIGHT TREND. 
    smoothing = 2 / (period + 1)
    ema = np.zeros_like(data) # NUMPY-ARRAY/COLUMN.
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = (data[i] * smoothing) + (ema[i-1] * (1 - smoothing))
    
    return ema
    
# FUNCTION TO CALCULATE TIME-PERIOD DATES.
def get_time_period_dates(period):

    end = datetime.now()
    if period == "1M":
        start = end - timedelta(days=30)
    elif period == "3M":
        start = end - timedelta(days=90)
    elif period == "6M":
        start = end - timedelta(days=180)
    elif period == "1Y":
        start = end - timedelta(days=365)
    elif period == "5Y":
        start = end - timedelta(days=1825)
    else:  # Max
        start = end - timedelta(days=3650)  # 10 years
        
    return start.date(), end.date()


# FUNCTION TO GENERATE_SIGNALS - BASED ON TECHNICAL INDICATORS.
@st.cache_data
def generate_signals(data):
    '''Generate trading signals based on technical indicators'''

    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    
    # SMA CROSS SIGNAL
    # NOTE: IF MA20 (SHORT-TERM) IS GREATER THAN MA50 (LONG-TERM) ASSIGN 1(INDICATING A BULLISH SIGNAL), OTHERWISE ASSIGN -1(INDICATING A BEARISH SIGNAL).
    signals['EMA_Signal'] = np.where(
        data['EMA12'] > data['EMA26'], 1, -1)
    
    # RSI SINGNALS - 1 --> BULLISH, -1 ---> BEARISH, 0 ---> NEUTRAL.
    signals['RSI_Signal'] = np.where(
        data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0)
    )
    
    # COMBINED SIGNALS.
    signals['Combined_Signal'] = signals['EMA_Signal'] + signals['RSI_Signal']

    signals['Trading_Signal'] = np.where(
        signals['Combined_Signal'] >= 1, 'Buy', np.where(signals['Combined_Signal'] <= -1, 'Sell', 'Neutral')
    )
    
    return signals

def fetch_sentiment_data(query="stock market"):
        '''Fetch And Analyze News Sentiment'''
        try:
            
            news_data = []
            
            # FETCH NEWS ARTICALES
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "language": "en",
                "pageSize": 10,
                "apiKey":   NEWS_API_KEY,
                "sortBy": "publishedAt"  # Sort by latest news
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                articles =  response.json()["articles"] 
                
            for article in articles:
                news_data.append({
                    'title': article['title'],
                    'description': article['description']
                })
                
            # ANALYZE SENTIMENT
            sia = SentimentIntensityAnalyzer()
            sentiments = []
            
            for article in news_data:
                # COMBINE TITLE AND DESCRIPTION FOR BETTER SENTIMENT ANALYSIS.
                text = f"{article['title']} {article['description']}"
                sentiment = sia.polarity_scores(text)
                
                # CALCULATE WEIGHTED SENTIMENT SCORE, CONSIDER POS/NEG SCORES TOO
                weighted_sentiment = (
                    sentiment['compound'] * 0.6 +  # MAIN SENTIMENT INDICATOR
                    (sentiment['pos'] - sentiment['neg']) * 0.4  # DIFFERENCE BETWEEN POSITIVE AND NEGATIVE.
                )
                
                sentiments.append(weighted_sentiment)
                
            # CALCULATE OVERALL SENTIMENT METRICS.
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                sentiment_std = np.std(sentiments)
                sentiment_momentum = np.mean(np.diff(sentiments)) if len(sentiments) > 1 else 0
                
                # COMBINE METERICS TO GET FINAL SENTIMENT SCORE.
                final_sentiment = (
                    avg_sentiment * 0.7 +  # BASE SENTIMENT
                    sentiment_momentum * 0.2 +  # TREND IN SENTIMENT
                    (1 - sentiment_std) * 0.1  # CONSISTENCY FACTOR
                )
                
                return final_sentiment
            
            return 0
                
        except Exception as e:
            print(f"Error fetching sentiment data: {e}")
            return 0
        
def format_changes(changes):
    '''Format the changes into a readable string with proper signs'''
    formatted_results = {}
    for period, values in changes.items():
        amount = values['amount_change']
        percent = values['percent_change']
        
        # Format with proper signs and currency symbol
        amount_str = f"${amount:+,.2f}"
        percent_str = f"{percent:+.2f}%"
        
        formatted_results[period] = {
            'amount_change': amount_str,
            'percent_change': percent_str
        }
        
    return formatted_results

def calculate_stock_changes(df):
    '''Calculate amount and percentage changes for stocks over different time periods.'''
    
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort DataFrame by date in ascending order
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    latest_price = df['Close'].iloc[-1]
    latest_date = df['Date'].iloc[-1]
    
    # Define time periods to analyze
    periods = {
        'Today': timedelta(days=1),
        '7 Days': timedelta(days=7),
        '30 Days': timedelta(days=30),
        '60 Days': timedelta(days=60),
    }
    
    results = {}
    
    for period_name, period_delta in periods.items():
        try:
            comparison_date = latest_date - period_delta
            
            # Find the closest price to comparison date
            mask = df['Date'] <= comparison_date
            if mask.any():  # Check if we have any dates matching our criteria
                comparison_price = df[mask].iloc[-1]['Close']
                
                # Calculate changes
                amount_change = latest_price - comparison_price
                percent_change = (amount_change / comparison_price) * 100
                
                results[period_name] = {
                    'amount_change': round(amount_change, 2),
                    'percent_change': round(percent_change, 2)
                }
            else:
                # Handle case where no matching dates are found
                results[period_name] = {
                    'amount_change': 0.0,
                    'percent_change': 0.0
                }
                print(f"Warning: No data available for {period_name} comparison")
                
        except Exception as e:
            results[period_name] = {
                'amount_change': 0.0,
                'percent_change': 0.0
            }
            print(f"Warning: Could not calculate {period_name} change: {str(e)}")
    
    # Format results once after all calculations
    formatted_results = format_changes(results)
    return formatted_results
    
if __name__ == '__main__':
    
    end = datetime.now()
    start = end - timedelta(days=365)
    df = load_data('BTC', start, end)
    
    print(df)