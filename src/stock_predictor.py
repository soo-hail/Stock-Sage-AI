# IMPORT NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
import requests
import warnings
import joblib
import nltk
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras_tuner as kt
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from data_augmentation import augment_stock_data 

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon')

from dotenv import load_dotenv
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY") 

class StockPredictor:
    def __init__(self, ticker, api_key_news):
        self.ticker = ticker
        self.news_api = NewsApiClient(api_key=api_key_news)
        self.scaler = MinMaxScaler() # TO NORMALIZE DATA
        self.close_scaler = MinMaxScaler() # SEPARETE SCALER FOR CLOSE-PRICE.
        self.model = None 
        self.seq_length = 60 # NO. OF TIME STEPS TO LOOK BACK(FOR PREDICTING CURRENT-PRICE) - CONSIDERED TO TRAIN MODEL - 2 MONTHS.
        self.features = ['Volume', 'SMA20', 'SMA50', 'SMA200', 'Volatility', 'MACD', 'RSI', 'ATR']
        
        
    def fetch_stock_data(self, years = 6):
        '''Fetch Historical Stock Data With Technical Indicators'''
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # FETCH STOCK DATA.
        df = yf.download(self.ticker, start=start_date, end=end_date)
        
        # FLATTEN MULTIINDEX COLUMNS IF THEY EXIST.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
        
        
        # TECHNICAL INDICATORS.
        # MOVING AVERAGE.
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # VOLATILITY - HOW MUCH THE PRICE OF A STOCK GOES UP AND DOWN OVER TIME.
        # IF PRICE CHANGE A LOT IN SHORT PERIOD ---> HIGH VOLATILE.
        # IF PRICE MOVES SLOWLY OR STAY STABLE ---> LESS VOLATILE.
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # AVERAGE TRUE RANGE (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
        
        return df.dropna()
    
    
    def fetch_sentiment_data(self, query="stock market"):
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
        
        
    def prepare_data(self, df):
        '''Prepare Data For LSTM Model'''
        
        
        target = df['Close'].values.reshape(-1, 1)

        # SCALE(NORMALIZE) CLOSE-PRIZE(TARGET)
        scaled_target = self.close_scaler.fit_transform(target)
        
        # SCALE(NORMALIZE) THE FEATURES.
        scaled_features = self.scaler.fit_transform(df[self.features])
        
        # CREATE TRAIN AND TEST DATA.
        X, y = [], []
        for i in range(self.seq_length, len(scaled_features)):
            X.append(scaled_features[i-self.seq_length:i])
            y.append(scaled_target[i])
        
        return np.array(X), np.array(y)
    
    
    def build_model(self, hp):
        '''Build LSTM Model With Adjusted Input Shape'''
        
        # TUNNABLE HYPER-PARAMETERS.
        hp_initial_units = hp.Int('initial_units', min_value=32, max_value=256, step=32)
        hp_layers = hp.Int('num_lstm_layers', min_value=1, max_value=3, step=1)
        hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=8)
        
        model = Sequential()
        
        # INPUT LAYER
        model.add(LSTM(
            units=hp_initial_units,
            return_sequences=hp_layers > 1,
            input_shape=(self.seq_length, len(self.features))
        ))
        model.add(Dropout(hp_dropout))
        
        # ADDITIONAL LSTM LAYERS
        for i in range(hp_layers - 1):
            units = hp_initial_units // (2 ** (i + 1)) # DECREASE UNITS IN SUBSEQUENT LAYERS.
            return_sequences = i < hp_layers - 2 # TRUE OR FALSE
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(hp_dropout))
            
        # DENSE LAYERS
        model.add(Dense(units=hp_dense_units, activation='relu'))
        model.add(Dense(units=1))
        
        # COMPILE THE MODEL.
        model.compile(
            optimizer=Adam(learning_rate=hp_learning_rate),
            loss='mean_squared_error'
        )
        
        return model
        
    
    def train_model(self, df, epochs=50, batch_size=32, max_trials=10):
        '''Train The LSTM Model'''
        
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = f'models/{self.ticker}_predictor_model.h5'
        best_model_path = f'models/{self.ticker}_best_model.h5' # TO SAVE BEST MODEL.
        scaler_path = f'models/{self.ticker}_feature_scaler.pkl'
        close_scaler_path = f'models/{self.ticker}_close_scaler.pkl'
        
        try:
            # IF MODEL EXISTS.
            if all(os.path.exists(p) for p in [best_model_path, scaler_path, close_scaler_path]):
                
                print("Loading existing best model...")
                self.model = load_model(best_model_path)
                self.scaler = joblib.load(scaler_path)
                self.close_scaler = joblib.load(close_scaler_path)
                
                return None, None
            
            print("Training new model...")  
            X, y = self.prepare_data(df)
            train_size = int(len(X) * 0.8)
            
            # SPLIT DATA
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # HYPER-PARAMETER TUNING
            tuner = kt.RandomSearch(
                self.build_model,
                objective='val_loss',
                max_trials=max_trials,
                directory='hyperparameter_tuning',
                project_name=f'{self.ticker}_stock_prediction',
                overwrite=True
            )
            
            # EARLY STOPPING CALLBACK.
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # SEARCH FOR BEST HYPER-PARAMETER.
            tuner.search(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # GET AND PRINT BEST HYPER-PARAMETER.
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print("Best Hyperparameters:")
            for param, value in best_hps.values.items():
                print(f"{param}: {value}")
            
            # BUILD AND TRAIN MODEL
            self.model = tuner.hypermodel.build(best_hps)
            
            # CREATE CHECK MODEL-CHECK-POINT CALLBACK.
            checkpoint = ModelCheckpoint(
                best_model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[checkpoint],
                verbose=2
            )
            
            # SAVE MODEL AND SCALES.
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.close_scaler, close_scaler_path)
            
            print(f"Model and scalers saved successfully for {self.ticker}.")

            # EVALUATE MODEL.
            self.model.load_weights(best_model_path)
            predictions = self.model.predict(X_test)
            self.evaluate_model(y_test, predictions)
            
            return history, (X_test, y_test)
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise
        
    def evaluate_model(self, y_test, predictions, show_plot=True):
        # COMPUTE REGRESSION METRICS.
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # MODEL ACCURACY.
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        accuracy = 100 - mape 
        
        print("\nModel Performance Metrics:")
        print("--------------------------")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Model Accuracy: {accuracy:.2f}%")
        
        # CREATE VISUALIZATION. 
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual', color='blue')
            plt.plot(predictions, label='Predicted', color='red', linestyle='--')
            plt.title('Stock Price: Actual vs Predicted')
            plt.xlabel('Time')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            plt.show()
          
            
    def predict_next_day_price(self, df):
        '''Predict Next Day's Price Using Only Feature Data'''

        # SELECT FEATURES
        features = ['Volume', 'SMA20', 'SMA50', 'SMA200', 'Volatility', 'MACD', 'RSI', 'ATR']

        # SCALE THE FEATURES
        last_sequence = df[features].tail(self.seq_length).values
        scaled_sequence = self.scaler.transform(last_sequence)
        
        # RESHAPE FOR PREDICTION.
        X_pred = scaled_sequence.reshape(1, self.seq_length, len(features))
        
        # MAKE PREDICTIONS AND INVERSE TRANSFORM.
        pred_scaled = self.model.predict(X_pred)
        prediction = self.close_scaler.inverse_transform(pred_scaled)
        
        return prediction[0][0]
    
    def predict_next_n_days(self, df, n_days=30):
        '''Predict stock prices for the next n days using iterative prediction'''

        # FEATURES TO BE CONSIDERED FOR PREDICTIONS.
        features = ['Volume', 'SMA20', 'SMA50', 'SMA200', 'Volatility', 'MACD', 'RSI', 'ATR']
        
        # INITIALIZE PREDICTIONS LIST WITH LAST KNOWN PRICE.
        predictions = [df['Close'].iloc[-1]]
        
        # CREATE A COPY.
        working_df = df.copy()
        
        for day in range(n_days):
            # GET THE LAST SEQUENCY OF DATA.
            last_sequence = working_df[features].tail(self.seq_length).values
            scaled_sequence = self.scaler.transform(last_sequence)
            
            # RESHAPE FOR PREDICTIONS.
            X_pred = scaled_sequence.reshape(1, self.seq_length, len(features))
            
            # MAKE PREDICTIONS AND REVERSE TRANSFORM.
            pred_scaled = self.model.predict(X_pred, verbose=0)
            prediction = self.close_scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(prediction)
            
            # UPDATE THE WORKING DIRECTORY WITH THE NEW PREDICTIONS.
            new_row = working_df.iloc[-1:].copy() # GET LAST ROW
            new_row.index = [working_df.index[-1] + pd.Timedelta(days=1)] # UPDATE TIME STAMP OF NEW-ROW, working_df.index[-1] ---> GIVES TIME-STAMP OF LAST ROW 
            new_row['Close'] = prediction # UPDATE CLOSE(STOCK-PRICE)

            # UPDATE TECHNICAL INDICATORS FOR NEW ROW.
            
            # DIALY_RETURNS
            new_row['Daily_Return'] = (prediction - working_df['Close'].iloc[-1]) / working_df['Close'].iloc[-1]
            
            # UPDATE MOVING AVERAGES
            for window in [20, 50, 200]:
                ma_name = f'SMA{window}'
                new_row[ma_name] = (working_df['Close'].iloc[-window+1:].sum() + prediction) / window # CALCULATES AVERAGE.
                
            # UPDATE VOLATILITY (20-DAYS STANDARD DEVIATION OF RETURNS)
            returns = pd.concat([working_df['Daily_Return'].tail(19), new_row['Daily_Return']]) # MERGES VALES AND RETURNS A ARRAY(COLUMN) 
            new_row['Volatility'] = returns.std()
            
            # UPDATE RSI
            delta = float(prediction) - float(working_df['Close'].iloc[-1])
            gain = max(delta, 0)
            loss = max(-delta, 0)
            avg_gain = (working_df['Close'].diff().clip(lower=0).tail(13).sum() + gain) / 14
            avg_loss = ((-working_df['Close'].diff().clip(upper=0).tail(13).sum()) + loss) / 14
            avg_loss = avg_loss.item() # CONVERT TO SCALER
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            new_row['RSI'] = 100 - (100 / (1 + rs))
            
            # UPDATE MACD
            exp1 = (working_df['Close'].tail(11).ewm(span=12, adjust=False).mean() * 11 + prediction) / 12
            exp2 = (working_df['Close'].tail(25).ewm(span=26, adjust=False).mean() * 25 + prediction) / 26
            new_row['MACD'] = exp1.iloc[-1] - exp2.iloc[-1]
            
            # Update ATR
            tr = max(
                abs(float(prediction) - float(working_df['Close'].iloc[-1])),
                abs(float(prediction) - float(working_df['Low'].iloc[-1])),
                abs(float(working_df['High'].iloc[-1]) - float(working_df['Low'].iloc[-1]))
            )
            
            # APPEND THE NEW ROW TO THE WORKING DATA-FRAME.
            working_df = pd.concat([working_df, new_row])
            
        return predictions[1:]
    
    
    def generate_trading_signal(self, current_price, predicted_price, sentiment):
        '''Generating Trading Signals Based On Prediction And Sentiment'''

        price_change_pct = (predicted_price - current_price) / current_price * 100
        
        # COMBINE PRICE PREDICTION WITH SENTIMENT.
        if price_change_pct > 1 and sentiment > 0.2:
            return "Strong Buy"
        elif price_change_pct > 0.5 or sentiment > 0.1:
            return "Buy"
        elif price_change_pct < -1 and sentiment < -0.2:
            return "Strong Sell"
        elif price_change_pct < -0.5 or sentiment < -0.1:
            return "Sell"
        else:
            return "Hold"
        
        
if __name__ == '__main__':
    
    predictor = StockPredictor("TSLA", NEWS_API_KEY)

    # FETCH DATA
    df = predictor.fetch_stock_data()
    print(df)

    # sentiment = predictor.fetch_sentiment_data()
    # print(sentiment)
    
    # TRAIN MODEL
    history, test_data = predictor.train_model(df)
    