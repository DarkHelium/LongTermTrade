import os
import json
from fin_r1_client import FinR1Client
from alpaca_trader import AlpacaTrader
from dotenv import load_dotenv

load_dotenv()

class ChatBot:
    def __init__(self):
        self.fin_r1 = FinR1Client()
        self.trader = None
        self.chat_log = []
        self.log_file = 'chat_log.json'
        self.is_live = False

    def start(self):
        print("Welcome to Fin-R1 Financial Advisor Chat.")
        self.is_live = input("Do you want to use live trading (requires bank linking)? (yes/no): ").lower() == 'yes'
        self.trader = AlpacaTrader(paper=not self.is_live)
        if self.is_live:
            print("For live trading, please link your bank account in the Alpaca app. Note: This is simulation for now.")
        else:
            print("Using paper trading (no real money).")

        while True:
            user_query = input("You: ")
            if user_query.lower() in ['exit', 'quit']:
                self.save_log()
                break
            self.chat_log.append({"role": "user", "content": user_query})

            prompt = f"User query: {user_query}. Provide truthful advice. Suggest stocks for short or long term if mentioned. Format suggestions as JSON: {{'stocks': [{'symbol': 'AAPL', 'qty': 1, 'term': 'long'}]}}"
            response = self.fin_r1.get_response(prompt)
            print(f"Fin-R1: {response}")
            self.chat_log.append({"role": "assistant", "content": response})

            try:
                suggestions = json.loads(response.split('JSON:')[1] if 'JSON:' in response else response)
                if 'stocks' in suggestions:
                    confirm = input("Do you want to invest in these suggestions? (yes/no): ").lower() == 'yes'
                    if confirm:
                        for stock in suggestions['stocks']:
                            self.trader.place_order(stock['symbol'], stock['qty'], 'buy')
                            print(f"Placed buy order for {stock['qty']} of {stock['symbol']}.")
            except:
                pass

    def save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.chat_log, f)
        print("Chat log saved.")

if __name__ == "__main__":
    bot = ChatBot()
    bot.start()