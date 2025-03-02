import os
import json
import pandas as pd
from datetime import datetime, timedelta

import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # FOR LLM

CRYPTOCARE_API_KEY = os.getenv("CRYPTOCARE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# INITIALIZE API'S
genai.configure(api_key=GEMINI_API_KEY)

class LLMAnalyzer:
    def __init__(self, symbol, data: pd.DataFrame):
        try:
            # VALIDATE INPUT
            required_columns = ['Close', 'High', 'Low', 'Volume', 'RSI', 'MA20', 'MA50', 'EMA12', 'EMA26']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Need: {required_columns}")
            
            if data.empty:
                raise ValueError("Input DataFrame is empty") 
            
            self.symbol = symbol
            self.current_price = round(float(data['Close'].iloc[-1]), 2)
            self.high = round(float(data['High'].iloc[-1]), 2)
            self.low = round(float(data['Low'].iloc[-1]), 2)
            self.volume = round(float(data['Volume'].iloc[-1]), 2)
            self.rsi = round(float(data['RSI'].iloc[-1]), 2)
            self.ma_20 = round(float(data['MA20'].iloc[-1]), 2)
            self.ma_50 = round(float(data['MA50'].iloc[-1]), 2)
            self.ema_12 = round(float(data['EMA12'].iloc[-1]), 2)
            self.ema_26 = round(float(data['EMA26'].iloc[-1]), 2)
            
             # DEFINE LLM - GEMINI PRO.
            # self.llm = ChatGoogleGenerativeAI(
            #     model="gemini-pro",
            #     google_api_key=GEMINI_API_KEY,
            #     temperature=0.7
            # )
            
            self.llm = ChatGroq(
                groq_api_key=os.getenv('GROQ_API_KEY'),
                model="llama-3.3-70b-versatile",
                temperature=0.1,
            )
            
        except KeyError as e:
            print(f"Missing column in DataFrame: {e}")
            raise
        except ValueError as e:
            print(f"Invalid input data: {e}")
            raise
        
        
        
    def generate_fallback_analysis(self) -> dict:
        """
        Generate a basic fallback analysis when the API call or parsing fails.
        """
        
        return {
            "market_points": [
                f"Basic price info: {self.symbol} is currently trading at ${self.current_price:,.2f}",
                "Volume analysis: Insufficient data for detailed analysis",
                "Technical indicators: Automated analysis currently unavailable"
            ],
            "outlook_points": [
                "Price target: Unable to generate detailed target at this time",
                f"Key levels: Observable range between ${self.low:,.2f} and ${self.high:,.2f}",
                "Trading strategy: Consider waiting for more stable market conditions"
            ],
            "support": self.low,
            "resistance": self.high
        }
    
    def get_technical_indicators_analysis(self, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                
                prompt = '''
                You are an expert stock analyst with deep knowledge of technical analysis. Analyze the technical indicators for {symbol} and provide a **detailed, insightful, and professional analysis** of trends, momentum, and potential market movements.

                Return a **valid** JSON object with a structured analysis in the exact JSON format below. Ensure that all values are properly formatted and that the analysis is **comprehensive, well-explained, and actionable** for traders and investors.
                

                {{
                    "Momentum Indicators": {{
                        "Relative Strength Index (RSI)": "RSI is {rsi}, indicating [Oversold / Overbought / Neutral]. Provide a detailed and precise explanation of what this means for the stock's momentum, including potential reversals or continuations. For example, if RSI is oversold, explain why this might signal a buying opportunity or if it could indicate further downside risk. Explanation in 3 lines"
                    }},

                    "Simple Moving Averages": {{
                        "Simple Moving Average (SMA)": "SMA-20 is {ma_20} and SMA-50 IS {ma_50}, indicating [Bullish crossover / Bearish crossover / Neutral]. Explain the significance of this crossover in detail, including how it reflects the stock's short-term vs. long-term trend. Discuss potential support/resistance levels and what traders should watch for in the coming days. Explanation in 3 lines"
                    }},
                    
                    "Exponential Moving Averages": {{
                        "Exponential Moving Average (EMA)": "EMA-12 is {ema_12} and EMA-26 is {ema_26} indicating [Bullish crossover / Bearish crossover / Neutral]. Provide a detailed analysis of the EMA crossover, including how it compares to the SMA crossover. Discuss the implications for trend strength and potential entry/exit points for traders. Explanation in 3 lines"
                    }},

                    "Volume Analysis": {{
                        "Volume": "Volume is [Higher / Lower / Normal] than average, indicating [Strong buying pressure / Weak momentum]. Explain the significance of the volume in detail, including how it confirms or contradicts the price action. Discuss whether the volume suggests accumulation, distribution, or a potential breakout."
                    }},

                    "Trend Analysis": {{
                        "Overall Trend": "[Bullish / Bearish / Neutral]. Provide a comprehensive summary of the overall trend, combining insights from RSI, SMA, EMA, and Volume. Discuss the likelihood of the trend continuing or reversing, and provide actionable advice for traders and investors based on the analysis."
                    }}
                }}
                '''
                
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = LLMChain(llm = self.llm, prompt=prompt)
                
                input_data = {
                    'symbol': self.symbol,
                    'rsi': self.rsi,
                    'volume': self.volume,
                    'ma_20': self.ma_20,
                    'ma_50': self.ma_50,
                    'ema_12': self.ema_12,
                    'ema_26': self.ema_26
                }
                
                # RUN THE CHAIN.
                analysis = chain.invoke(input_data)
                return analysis
            except Exception as e:
                print(f'API call error: {e}')
                if attempt == max_retries - 1:
                    return self.generate_fallback_analysis() # PENDING ATTRIBUTES.
                
        return self.generate_fallback_analysis() # PENDING ATTRIBUTES.
                
        
if __name__ == '__main__':
    # ticker = 'AAPL'
    # end = datetime.now()
    # start = end - timedelta(days=365)

    # data = load_data(ticker, start, end)
    # analyzer = LLMAnalyzer(ticker, data)
    
    # analysis = analyzer.get_technical_indicators_analysis()

    # print(analysis)
    pass
    
    