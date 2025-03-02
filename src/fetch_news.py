import os
import requests
import streamlit as st

from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_stock_news(query="stock market", language="en", page_size=10):
    """
    Fetch live stock news using NewsAPI.
    """

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
        "sortBy": "publishedAt"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        print(f"❌ Error fetching news: {response.status_code}, Message: {response.text}")
        return []


if __name__ == '__main__':
    pass