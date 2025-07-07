import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from alpaca_trade_api import REST
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Alpaca API credentials
API_KEY="PKLYOCBPAMR5M8DPJM9G"
API_SECRET="Mtaa4I06zsvvxt92MOwg8B0WZa80cOMLjUcCjztw"

BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDENTIALS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

broker = Alpaca(ALPACA_CREDENTIALS)

# Initialize sentiment analysis model
finbert = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')

class LongTermBot(Strategy):
    def initialize(self, symbol="SPY", weekly_investment=1000, sentiment_multiplier=2.0):
        self.symbol = symbol
        self.weekly_investment = weekly_investment
        self.sentiment_multiplier = sentiment_multiplier
        self.last_investment_week = -1
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

        self.log_message(f"Long-term investment bot initialized for {self.symbol} with a base weekly investment of ${self.weekly_investment}")

    def get_news_sentiment(self):
        # Fetch news from the last 7 days
        today = self.get_datetime()
        seven_days_ago = today - timedelta(days=7)
        
        news = self.api.get_news(symbol=self.symbol, 
                                 start=seven_days_ago.strftime('%Y-%m-%dT%H:%M:%SZ'), 
                                 end=today.strftime('%Y-%m-%dT%H:%M:%SZ'))
        headlines = [item.headline for item in news]

        if not headlines:
            return 0.0 # Neutral sentiment if no news

        # Analyze sentiment
        inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
        outputs = finbert(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Aggregate sentiment (positive - negative)
        sentiment_score = predictions[:, 0].mean() - predictions[:, 1].mean()
        return sentiment_score.item()

    def on_trading_iteration(self):
        current_date = self.datetime.date()
        current_week = current_date.isocalendar()[1]

        # Invest once per week
        if current_week != self.last_investment_week:
            self.last_investment_week = current_week
            
            # Get sentiment
            sentiment = self.get_news_sentiment()
            self.log_message(f"Current market sentiment for {self.symbol}: {sentiment:.2f}")

            # Adjust investment based on sentiment
            investment_amount = self.weekly_investment
            if sentiment < -0.3: # Strong negative sentiment
                investment_amount *= self.sentiment_multiplier
                self.log_message(f"Strong negative sentiment detected. Increasing investment to ${investment_amount:.2f}")

            # Dollar-Cost Averaging
            cash = self.get_cash()
            amount_to_invest = min(investment_amount, cash)
            current_price = self.get_last_price(self.symbol)

            if amount_to_invest > 0 and current_price > 0:
                shares = amount_to_invest / current_price
                self.buy(symbol=self.symbol, quantity=shares)
                self.log_message(f"Weekly investment: Bought {shares:.2f} shares of {self.symbol} at ${current_price:.2f}")

# For backtesting
if __name__ == "__main__":
    # Define backtest parameters
    backtesting_start = datetime(2022, 1, 1)
    backtesting_end = datetime(2024, 6, 25)
    
    # Run the backtest
    LongTermBot.run_backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        parameters={"symbol": "SPY", "weekly_investment": 1000, "sentiment_multiplier": 1.5}
    )







