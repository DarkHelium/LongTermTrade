import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

from pathlib import Path

# Load .env file from the backend directory
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

class AlpacaTrader:
    def __init__(self, paper=True):
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError("Alpaca API keys not found in environment variables.")
        self.paper = paper  # Store paper trading mode as instance attribute
        self.client = TradingClient(api_key, api_secret, paper=paper)

    def get_account(self):
        return self.client.get_account()

    def place_order(self, symbol: str, qty: float, side: str):
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        return self.client.submit_order(order_data=order_data)

    def check_buying_power(self):
        account = self.get_account()
        return float(account.buying_power)