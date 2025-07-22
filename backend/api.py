from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from fin_r1_client import FinR1Client
from alpaca_trader import AlpacaTrader
from dotenv import load_dotenv
from pydantic import BaseModel

from pathlib import Path

# Load .env file from the backend directory
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:4000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="../frontend/dist/frontend/browser"), name="static")

templates = Jinja2Templates(directory="../frontend/dist/frontend/browser")

class ChatRequest(BaseModel):
    query: str
    is_live: bool = False

fin_r1 = FinR1Client()
trader = None  # Will be initialized per request or session
chat_log = []  # Simple in-memory log; use database for production

@app.on_event("startup")
def startup():
    global trader
    # For simplicity, initialize here; in production, handle per user
    trader = AlpacaTrader(paper=True)  # Default to paper

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    global chat_log, trader
    user_query = chat_request.query
    is_live = chat_request.is_live
    
    try:
        # Check if query is investment-related
        investment_keywords = ['buy', 'sell', 'invest', 'stock', 'portfolio', 'trade', 'purchase']
        is_investment_query = any(keyword in user_query.lower() for keyword in investment_keywords)
        
        # Only initialize trader for investment-related queries
        if is_investment_query:
            if trader is None or is_live != (not trader.paper):
                trader = AlpacaTrader(paper=not is_live)
        
        chat_log.append({"role": "user", "content": user_query})
        
        # Create appropriate prompt based on query type
        if is_investment_query:
            prompt = f"User query: {user_query}. Provide truthful investment advice. If suggesting specific stocks, format as JSON: {{\"stocks\": [{{\"symbol\": \"AAPL\", \"qty\": 1, \"term\": \"long\"}}]}}"
        else:
            prompt = f"User query: {user_query}. Provide helpful and truthful information about financial markets, trends, or general financial advice. Do not suggest specific stock purchases."
        
        # Get response from Fin-R1 model with error handling
        try:
            response = fin_r1.get_response(prompt)
        except Exception as e:
            print(f"Fin-R1 Error: {str(e)}")  # Log for debugging
            if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                error_msg = "Authentication error: Please check your ANTHROPIC_API_KEY in the .env file."
            elif "rate limit" in str(e).lower():
                error_msg = "Rate limit exceeded. Please try again in a moment."
            elif "model" in str(e).lower():
                error_msg = "Model error: The specified Claude model may not be available."
            else:
                error_msg = f"Error connecting to Anthropic API: {str(e)}"
            return {"response": error_msg, "suggestions": []}
        
        chat_log.append({"role": "assistant", "content": response})
        
        # Parse suggestions from response only for investment queries
        suggestions = []
        if is_investment_query:
            try:
                if 'JSON:' in response:
                    suggestions_str = response.split('JSON:')[1].strip()
                    suggestions_data = json.loads(suggestions_str)
                    suggestions = suggestions_data.get('stocks', [])
                elif '{' in response and '}' in response:
                    # Try to extract JSON from response
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start != -1 and end != 0:
                        suggestions_str = response[start:end]
                        suggestions_data = json.loads(suggestions_str)
                        suggestions = suggestions_data.get('stocks', [])
            except json.JSONDecodeError:
                pass  # No valid JSON found, return empty suggestions
        
        return {"response": f"\n{response}\n", "suggestions": suggestions}
            
    except Exception as e:
        print(f"General API Error: {str(e)}")  # Log for debugging
        error_msg = f"Sorry, I encountered an error: {str(e)}. Please check the server logs for more details."
        return {"response": error_msg, "suggestions": []}

@app.post("/invest")
async def invest(investments: list[dict]):
    global trader
    results = []
    for stock in investments:
        try:
            order = trader.place_order(stock['symbol'], stock['qty'], 'buy')
            results.append({"symbol": stock['symbol'], "qty": stock['qty'], "status": "placed"})
        except Exception as e:
            results.append({"symbol": stock['symbol'], "qty": stock['qty'], "status": str(e)})
    return {"results": results}

@app.get("/log")
def get_log():
    return chat_log