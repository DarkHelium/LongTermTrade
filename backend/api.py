from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from fin_r1_client import FinR1Client
from alpaca_trader import AlpacaTrader
from cnbc_analyzer import CNBCAnalyzer
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

class CNBCAnalysisRequest(BaseModel):
    days_back: int = 7
    min_frequency: int = 3

fin_r1 = FinR1Client()
cnbc_analyzer = CNBCAnalyzer()
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
        cnbc_keywords = ['cnbc', 'contrarian', 'sentiment', 'media', 'news', 'pattern', 'opposite']
        
        is_investment_query = any(keyword in user_query.lower() for keyword in investment_keywords)
        is_cnbc_query = any(keyword in user_query.lower() for keyword in cnbc_keywords)
        
        # Only initialize trader for investment-related queries
        if is_investment_query:
            if trader is None or is_live != (not trader.paper):
                trader = AlpacaTrader(paper=not is_live)
        
        chat_log.append({"role": "user", "content": user_query})
        
        # Create appropriate prompt based on query type
        if is_cnbc_query or 'contrarian' in user_query.lower():
            # Get CNBC analysis for contrarian queries
            try:
                cnbc_analysis = cnbc_analyzer.get_contrarian_recommendations(days_back=7, min_frequency=2)
                
                # Create enhanced prompt with CNBC data
                prompt = f"""User query: {user_query}
                
CNBC Contrarian Analysis Data:
- Articles analyzed: {cnbc_analysis['articles_analyzed']}
- Patterns found: {cnbc_analysis['patterns_found']}
- Buy signals: {len(cnbc_analysis['buy_signals'])}
- Sell signals: {len(cnbc_analysis['sell_signals'])}

Top Buy Signals (Contrarian): {cnbc_analysis['buy_signals'][:3]}
Top Sell Signals (Contrarian): {cnbc_analysis['sell_signals'][:3]}

Based on this CNBC sentiment analysis showing repetitive patterns, provide contrarian investment advice. 
Explain how media repetition often signals market extremes and why doing the opposite can be profitable.
If suggesting specific stocks, format as JSON: {{"stocks": [{{"symbol": "AAPL", "qty": 1, "term": "long"}}]}}"""
                
            except Exception as e:
                prompt = f"User query: {user_query}. Provide contrarian investment advice based on general market sentiment principles. Note: CNBC analysis temporarily unavailable."
                
        elif is_investment_query:
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

@app.post("/cnbc-analysis")
async def analyze_cnbc_patterns(request: CNBCAnalysisRequest):
    """Analyze CNBC articles for repetitive patterns and provide contrarian signals"""
    try:
        analysis = cnbc_analyzer.get_contrarian_recommendations(
            days_back=request.days_back,
            min_frequency=request.min_frequency
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing CNBC patterns: {str(e)}")

@app.get("/cnbc-quick-analysis")
async def quick_cnbc_analysis():
    """Quick CNBC analysis with default parameters"""
    try:
        analysis = cnbc_analyzer.get_contrarian_recommendations(days_back=3, min_frequency=2)
        return {
            "quick_summary": {
                "total_patterns": analysis["patterns_found"],
                "buy_signals": len(analysis["buy_signals"]),
                "sell_signals": len(analysis["sell_signals"]),
                "articles_analyzed": analysis["articles_analyzed"]
            },
            "top_buy_signals": analysis["buy_signals"][:3],
            "top_sell_signals": analysis["sell_signals"][:3],
            "analysis_date": analysis["analysis_date"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in quick CNBC analysis: {str(e)}")

@app.get("/cnbc-stock/{symbol}")
async def analyze_specific_stock(symbol: str, days_back: int = 7):
    """Analyze CNBC coverage for a specific stock symbol"""
    try:
        # Get full analysis
        analysis = cnbc_analyzer.get_contrarian_recommendations(days_back=days_back, min_frequency=1)
        
        # Filter for specific stock
        stock_analysis = {
            "symbol": symbol.upper(),
            "buy_signals": [s for s in analysis["buy_signals"] if s["stock_symbol"] == symbol.upper()],
            "sell_signals": [s for s in analysis["sell_signals"] if s["stock_symbol"] == symbol.upper()],
            "hold_signals": [s for s in analysis["hold_signals"] if s["stock_symbol"] == symbol.upper()],
            "analysis_date": analysis["analysis_date"],
            "time_period": f"Last {days_back} days"
        }
        
        if not any([stock_analysis["buy_signals"], stock_analysis["sell_signals"], stock_analysis["hold_signals"]]):
            return {
                "symbol": symbol.upper(),
                "message": f"No significant CNBC coverage found for {symbol.upper()} in the last {days_back} days",
                "analysis_date": analysis["analysis_date"]
            }
        
        return stock_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {symbol}: {str(e)}")