from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv
import anthropic
from agent_system import create_buffett_agent
from market_data_service import MarketDataService
from finnhub_service import FinnhubService, FinnhubStock, FinnhubQuote

# Load environment variables
load_dotenv()

app = FastAPI(title="Warren Buffett Stock Analysis API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
anthropic_client = None
if os.getenv("ANTHROPIC_API_KEY"):
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize AI Agent System
buffett_agent = create_buffett_agent(os.getenv("ANTHROPIC_API_KEY"))

# Initialize Finnhub service
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
if not finnhub_api_key:
    raise ValueError("FINNHUB_API_KEY environment variable not set")
finnhub_service = FinnhubService(api_key=finnhub_api_key)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    timestamp: Optional[datetime] = None

class ChatResponse(BaseModel):
    response: str
    buffett_picks: Optional[List[str]] = None
    timestamp: datetime
    reasoning_traces: Optional[List[str]] = None
    tool_usage: Optional[List[Dict[str, Any]]] = None
    agent_context: Optional[str] = None

class AgentChatMessage(BaseModel):
    message: str
    use_agent: bool = True
    timestamp: Optional[datetime] = None

class AgentChatResponse(BaseModel):
    response: str
    reasoning_traces: List[str]
    tool_usage: List[Dict[str, Any]]
    iterations: int
    context: str
    buffett_picks: Optional[List[str]] = None
    timestamp: datetime

class StockInfo(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

class BuffettScore(BaseModel):
    symbol: str
    overall_score: float
    criteria_scores: Dict[str, float]
    analysis: str
    recommendation: str

class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    current_price: float
    buffett_score: BuffettScore
    financial_metrics: Dict[str, Any]
    competitive_advantages: List[str]
    risks: List[str]

# Warren Buffett's top stock picks (updated regularly)
BUFFETT_TOP_PICKS = [
    "AAPL",  # Apple Inc.
    "BRK-B", # Berkshire Hathaway
    "KO",    # Coca-Cola
    "BAC",   # Bank of America
    "CVX",   # Chevron
    "OXY",   # Occidental Petroleum
    "MCO",   # Moody's Corporation
    "AXP",   # American Express
    "KHC",   # Kraft Heinz
    "VZ",    # Verizon
]

class BuffettAnalyzer:
    """Warren Buffett-style stock analysis engine"""
    
    def __init__(self):
        self.criteria_weights = {
            "economic_moat": 0.25,
            "financial_strength": 0.20,
            "management_quality": 0.15,
            "valuation": 0.20,
            "growth_prospects": 0.10,
            "dividend_consistency": 0.10
        }
    
    def analyze_stock(self, symbol: str) -> BuffettScore:
        """Analyze a stock using Warren Buffett's criteria"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            
            # Calculate individual criteria scores
            scores = {}
            scores["economic_moat"] = self._analyze_economic_moat(info, financials)
            scores["financial_strength"] = self._analyze_financial_strength(info, balance_sheet)
            scores["management_quality"] = self._analyze_management_quality(info)
            scores["valuation"] = self._analyze_valuation(info)
            scores["growth_prospects"] = self._analyze_growth_prospects(info, financials)
            scores["dividend_consistency"] = self._analyze_dividend_consistency(stock)
            
            # Calculate weighted overall score
            overall_score = sum(scores[criteria] * self.criteria_weights[criteria] 
                              for criteria in scores)
            
            # Generate analysis and recommendation
            analysis = self._generate_analysis(symbol, scores, info)
            recommendation = self._generate_recommendation(overall_score)
            
            return BuffettScore(
                symbol=symbol,
                overall_score=round(overall_score, 2),
                criteria_scores=scores,
                analysis=analysis,
                recommendation=recommendation
            )
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error analyzing {symbol}: {str(e)}")
    
    def _analyze_economic_moat(self, info: dict, financials: pd.DataFrame) -> float:
        """Analyze economic moat strength (0-100)"""
        score = 50  # Base score
        
        # Brand strength (market cap relative to revenue)
        market_cap = info.get('marketCap', 0)
        revenue = info.get('totalRevenue', 1)
        if market_cap and revenue:
            brand_multiple = market_cap / revenue
            if brand_multiple > 10:
                score += 20
            elif brand_multiple > 5:
                score += 10
        
        # Profit margins (higher margins suggest pricing power)
        profit_margin = info.get('profitMargins', 0)
        if profit_margin > 0.20:
            score += 15
        elif profit_margin > 0.10:
            score += 10
        elif profit_margin > 0.05:
            score += 5
        
        # Return on equity (efficient use of shareholder equity)
        roe = info.get('returnOnEquity', 0)
        if roe > 0.20:
            score += 15
        elif roe > 0.15:
            score += 10
        elif roe > 0.10:
            score += 5
        
        return min(100, max(0, score))
    
    def _analyze_financial_strength(self, info: dict, balance_sheet: pd.DataFrame) -> float:
        """Analyze financial strength (0-100)"""
        score = 50  # Base score
        
        # Debt-to-equity ratio (lower is better)
        debt_to_equity = info.get('debtToEquity', 100)
        if debt_to_equity < 30:
            score += 20
        elif debt_to_equity < 50:
            score += 15
        elif debt_to_equity < 100:
            score += 10
        else:
            score -= 10
        
        # Current ratio (liquidity)
        current_ratio = info.get('currentRatio', 1)
        if current_ratio > 2:
            score += 15
        elif current_ratio > 1.5:
            score += 10
        elif current_ratio > 1:
            score += 5
        else:
            score -= 10
        
        # Free cash flow
        free_cash_flow = info.get('freeCashflow', 0)
        if free_cash_flow > 0:
            score += 15
        
        return min(100, max(0, score))
    
    def _analyze_management_quality(self, info: dict) -> float:
        """Analyze management quality (0-100)"""
        score = 50  # Base score
        
        # Return on assets (management efficiency)
        roa = info.get('returnOnAssets', 0)
        if roa > 0.10:
            score += 20
        elif roa > 0.05:
            score += 15
        elif roa > 0.02:
            score += 10
        
        # Revenue growth (management's ability to grow business)
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth > 0.10:
            score += 15
        elif revenue_growth > 0.05:
            score += 10
        elif revenue_growth > 0:
            score += 5
        
        # Earnings growth
        earnings_growth = info.get('earningsGrowth', 0)
        if earnings_growth > 0.10:
            score += 15
        elif earnings_growth > 0.05:
            score += 10
        elif earnings_growth > 0:
            score += 5
        
        return min(100, max(0, score))
    
    def _analyze_valuation(self, info: dict) -> float:
        """Analyze valuation attractiveness (0-100)"""
        score = 50  # Base score
        
        # P/E ratio (lower is generally better for value)
        pe_ratio = info.get('trailingPE', 50)
        if pe_ratio < 15:
            score += 20
        elif pe_ratio < 20:
            score += 15
        elif pe_ratio < 25:
            score += 10
        elif pe_ratio < 30:
            score += 5
        else:
            score -= 10
        
        # Price-to-book ratio
        pb_ratio = info.get('priceToBook', 5)
        if pb_ratio < 1.5:
            score += 15
        elif pb_ratio < 2.5:
            score += 10
        elif pb_ratio < 3.5:
            score += 5
        
        # PEG ratio (P/E relative to growth)
        peg_ratio = info.get('pegRatio', 2)
        if peg_ratio and peg_ratio < 1:
            score += 15
        elif peg_ratio and peg_ratio < 1.5:
            score += 10
        
        return min(100, max(0, score))
    
    def _analyze_growth_prospects(self, info: dict, financials: pd.DataFrame) -> float:
        """Analyze growth prospects (0-100)"""
        score = 50  # Base score
        
        # Revenue growth trend
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth > 0.15:
            score += 20
        elif revenue_growth > 0.10:
            score += 15
        elif revenue_growth > 0.05:
            score += 10
        elif revenue_growth > 0:
            score += 5
        
        # Market position
        market_cap = info.get('marketCap', 0)
        if market_cap > 100_000_000_000:  # $100B+
            score += 15
        elif market_cap > 10_000_000_000:  # $10B+
            score += 10
        elif market_cap > 1_000_000_000:   # $1B+
            score += 5
        
        return min(100, max(0, score))
    
    def _analyze_dividend_consistency(self, stock) -> float:
        """Analyze dividend consistency (0-100)"""
        score = 50  # Base score
        
        try:
            dividends = stock.dividends
            if len(dividends) > 0:
                # Has dividends
                score += 20
                
                # Dividend growth
                recent_dividends = dividends.tail(5)
                if len(recent_dividends) >= 2:
                    if recent_dividends.iloc[-1] > recent_dividends.iloc[0]:
                        score += 15  # Growing dividends
                    elif recent_dividends.iloc[-1] == recent_dividends.iloc[0]:
                        score += 10  # Stable dividends
                
                # Dividend yield
                dividend_yield = stock.info.get('dividendYield', 0)
                if dividend_yield and dividend_yield > 0.03:
                    score += 15
                elif dividend_yield and dividend_yield > 0.02:
                    score += 10
            else:
                # No dividends, but could be growth stock
                score = 40
        except:
            score = 40
        
        return min(100, max(0, score))
    
    def _generate_analysis(self, symbol: str, scores: dict, info: dict) -> str:
        """Generate Warren Buffett-style analysis"""
        company_name = info.get('longName', symbol)
        
        analysis = f"Analysis of {company_name} ({symbol}) through Warren Buffett's lens:\n\n"
        
        # Economic moat analysis
        moat_score = scores['economic_moat']
        if moat_score >= 80:
            analysis += "🏰 STRONG ECONOMIC MOAT: This company demonstrates exceptional competitive advantages. "
        elif moat_score >= 60:
            analysis += "🛡️ MODERATE ECONOMIC MOAT: The company has some competitive advantages. "
        else:
            analysis += "⚠️ WEAK ECONOMIC MOAT: Limited competitive advantages observed. "
        
        # Financial strength
        financial_score = scores['financial_strength']
        if financial_score >= 80:
            analysis += "The balance sheet is fortress-like with minimal debt and strong cash position. "
        elif financial_score >= 60:
            analysis += "Solid financial foundation with manageable debt levels. "
        else:
            analysis += "Financial position requires careful monitoring. "
        
        # Valuation assessment
        valuation_score = scores['valuation']
        if valuation_score >= 80:
            analysis += "Currently trading at an attractive valuation - a potential 'wonderful company at a fair price'. "
        elif valuation_score >= 60:
            analysis += "Reasonably valued given the company's fundamentals. "
        else:
            analysis += "Appears overvalued at current levels. "
        
        # Management quality
        mgmt_score = scores['management_quality']
        if mgmt_score >= 80:
            analysis += "Management demonstrates exceptional capital allocation skills. "
        elif mgmt_score >= 60:
            analysis += "Competent management team with decent track record. "
        else:
            analysis += "Management effectiveness is questionable. "
        
        return analysis
    
    def _generate_recommendation(self, overall_score: float) -> str:
        """Generate investment recommendation"""
        if overall_score >= 85:
            return "STRONG BUY - Exceptional Buffett-style investment opportunity"
        elif overall_score >= 75:
            return "BUY - Good value investment with solid fundamentals"
        elif overall_score >= 65:
            return "HOLD - Decent company but wait for better entry point"
        elif overall_score >= 50:
            return "WEAK HOLD - Below average investment opportunity"
        else:
            return "AVOID - Does not meet Buffett's investment criteria"

# Initialize services
buffett_analyzer = BuffettAnalyzer()
market_data_service = MarketDataService()

@app.get("/")
async def root():
    return {"message": "Warren Buffett Stock Analysis API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/buffett-picks", response_model=List[StockInfo])
async def get_buffett_picks():
    """Get Warren Buffett's current top stock picks with real-time data"""
    stocks = []
    
    try:
        # Use the market data service with caching to avoid rate limiting
        quotes_result = await market_data_service.get_multiple_quotes(BUFFETT_TOP_PICKS)
        
        if quotes_result['status'] == 'success':
            for symbol, quote_data in quotes_result['quotes'].items():
                if quote_data['status'] == 'success':
                    stock_info = StockInfo(
                        symbol=symbol,
                        name=quote_data.get('company_name', symbol),
                        price=quote_data.get('price', 0),
                        change=quote_data.get('change', 0),
                        change_percent=quote_data.get('change_percent', 0),
                        market_cap=quote_data.get('market_cap'),
                        pe_ratio=quote_data.get('pe_ratio'),
                        dividend_yield=quote_data.get('dividend_yield')
                    )
                    stocks.append(stock_info)
                else:
                    print(f"Error fetching data for {symbol}: {quote_data.get('error', 'Unknown error')}")
        else:
            print(f"Error fetching multiple quotes: {quotes_result.get('error', 'Unknown error')}")
            
        # If we have no stocks from the service, fall back to a simple approach
        if not stocks:
            for symbol in BUFFETT_TOP_PICKS[:3]:  # Limit to 3 to avoid rate limiting
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    stock_info = StockInfo(
                        symbol=symbol,
                        name=info.get('longName', symbol),
                        price=info.get('currentPrice', 0),
                        change=0,  # Simplified for fallback
                        change_percent=0,  # Simplified for fallback
                        market_cap=info.get('marketCap'),
                        pe_ratio=info.get('trailingPE'),
                        dividend_yield=info.get('dividendYield')
                    )
                    stocks.append(stock_info)
                    
                except Exception as e:
                    print(f"Fallback error for {symbol}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error in get_buffett_picks: {e}")
        # Return empty list if all fails
        return []
    
    return stocks

@app.get("/stock/{symbol}/analysis", response_model=StockAnalysis)
async def analyze_stock(symbol: str):
    """Get comprehensive Warren Buffett-style analysis for a specific stock"""
    try:
        # Get basic stock info
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        # Get Buffett score
        buffett_score = buffett_analyzer.analyze_stock(symbol.upper())
        
        # Get current price
        hist = ticker.history(period="1d")
        current_price = hist['Close'].iloc[-1] if len(hist) > 0 else info.get('currentPrice', 0)
        
        # Extract financial metrics
        financial_metrics = {
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'),
            "pb_ratio": info.get('priceToBook'),
            "debt_to_equity": info.get('debtToEquity'),
            "roe": info.get('returnOnEquity'),
            "roa": info.get('returnOnAssets'),
            "profit_margin": info.get('profitMargins'),
            "revenue_growth": info.get('revenueGrowth'),
            "dividend_yield": info.get('dividendYield'),
            "free_cash_flow": info.get('freeCashflow')
        }
        
        # Identify competitive advantages
        competitive_advantages = []
        if buffett_score.criteria_scores.get('economic_moat', 0) >= 70:
            competitive_advantages.extend([
                "Strong brand recognition and customer loyalty",
                "Significant barriers to entry in the industry",
                "Pricing power due to market position"
            ])
        
        if info.get('profitMargins', 0) > 0.15:
            competitive_advantages.append("High profit margins indicating pricing power")
        
        if info.get('returnOnEquity', 0) > 0.15:
            competitive_advantages.append("Efficient use of shareholder capital")
        
        # Identify risks
        risks = []
        if info.get('debtToEquity', 0) > 100:
            risks.append("High debt levels may limit financial flexibility")
        
        if info.get('trailingPE', 0) > 30:
            risks.append("High valuation may limit upside potential")
        
        if buffett_score.criteria_scores.get('economic_moat', 0) < 50:
            risks.append("Limited competitive advantages in a competitive industry")
        
        return StockAnalysis(
            symbol=symbol.upper(),
            company_name=info.get('longName', symbol),
            current_price=round(current_price, 2),
            buffett_score=buffett_score,
            financial_metrics=financial_metrics,
            competitive_advantages=competitive_advantages,
            risks=risks
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing {symbol}: {str(e)}")

@app.get("/stock/{symbol}/score", response_model=BuffettScore)
async def get_buffett_score(symbol: str):
    """Get Warren Buffett score for a specific stock"""
    return buffett_analyzer.analyze_stock(symbol.upper())

@app.post("/chat", response_model=ChatResponse)
async def chat_with_buffett(message: ChatMessage):
    """Enhanced chat with Warren Buffett AI using ReAct agent system"""
    
    user_message = message.message
    
    # Use the ReAct agent for intelligent responses
    try:
        agent_result = await buffett_agent.process_query(user_message)
        
        # Check if the user is asking for stock recommendations
        buffett_picks = None
        if any(keyword in user_message.lower() for keyword in ["pick", "recommend", "suggest", "stock", "buy"]):
            buffett_picks = BUFFETT_TOP_PICKS[:5]
        
        return ChatResponse(
            response=agent_result["response"],
            buffett_picks=buffett_picks,
            timestamp=datetime.now(),
            reasoning_traces=agent_result.get("reasoning_traces", []),
            tool_usage=agent_result.get("tool_usage", []),
            agent_context=agent_result.get("context", "")
        )
        
    except Exception as e:
        print(f"Error with agent system: {e}")
        
        # Fallback to simple response if agent fails
        if "pick" in user_message.lower() or "recommend" in user_message.lower():
            response = """Based on my investment philosophy, here are my current top picks:

1. **Apple (AAPL)** - Incredible brand loyalty and ecosystem. It's like owning a toll bridge that everyone must cross.

2. **Coca-Cola (KO)** - I've owned this for decades. People will drink Coke in good times and bad. It's a wonderful business.

3. **Bank of America (BAC)** - Well-managed bank with improving efficiency. Banking is a good business when done right.

4. **Chevron (CVX)** - Energy will always be needed. They have excellent management and strong cash flows.

5. **American Express (AXP)** - The wealthy use Amex, and the wealthy keep getting wealthier. Simple as that.

Remember, buy businesses, not stocks. These companies have economic moats that protect them from competition."""
            
            return ChatResponse(
                response=response,
                buffett_picks=BUFFETT_TOP_PICKS[:5],
                timestamp=datetime.now()
            )
        else:
            response = """Remember my key principles: invest in businesses you understand, look for companies with 
            competitive advantages (economic moats), buy at reasonable prices, and think long-term. The stock market 
            is a voting machine in the short run, but a weighing machine in the long run."""
            
            return ChatResponse(
                response=response,
                timestamp=datetime.now()
            )

@app.post("/agent-chat", response_model=AgentChatResponse)
async def advanced_agent_chat(message: AgentChatMessage):
    """Advanced chat endpoint with full agent transparency and reasoning traces"""
    
    try:
        agent_result = await buffett_agent.process_query(message.message)
        
        # Check if the user is asking for stock recommendations
        buffett_picks = None
        if any(keyword in message.message.lower() for keyword in ["pick", "recommend", "suggest", "stock", "buy"]):
            buffett_picks = BUFFETT_TOP_PICKS[:5]
        
        return AgentChatResponse(
            response=agent_result["response"],
            reasoning_traces=agent_result["reasoning_traces"],
            tool_usage=agent_result["tool_usage"],
            iterations=agent_result["iterations"],
            context=agent_result["context"],
            buffett_picks=buffett_picks,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        return AgentChatResponse(
            response=f"I apologize, but I encountered an error: {str(e)}. Let me provide some general investment wisdom instead: Focus on businesses with strong competitive moats, excellent management, and reasonable valuations.",
            reasoning_traces=[f"Error occurred: {str(e)}"],
            tool_usage=[],
            iterations=0,
            context="Error state",
            timestamp=datetime.now()
        )

@app.get("/stocks/all")
async def get_all_stocks():
    """Get all available US stocks from Finnhub"""
    try:
        result = await finnhub_service.get_all_us_stocks()
        if result['status'] == 'success':
            return {
                'status': 'success',
                'stocks': result['stocks'][:100],  # Limit to first 100 for performance
                'total_count': result['count']
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stocks: {str(e)}")

@app.get("/stocks/popular")
async def get_popular_stocks():
    """Get popular/trending stocks with quotes"""
    try:
        result = await finnhub_service.get_popular_stocks(limit=20)
        if result['status'] == 'success':
            # Add company descriptions for popular stocks
            popular_descriptions = {
                'AAPL': 'Apple Inc. - Technology hardware and software',
                'MSFT': 'Microsoft Corporation - Software and cloud services',
                'GOOGL': 'Alphabet Inc. - Internet search and advertising',
                'AMZN': 'Amazon.com Inc. - E-commerce and cloud computing',
                'TSLA': 'Tesla Inc. - Electric vehicles and energy',
                'META': 'Meta Platforms Inc. - Social media and metaverse',
                'NVDA': 'NVIDIA Corporation - Graphics and AI chips',
                'NFLX': 'Netflix Inc. - Streaming entertainment',
                'BRK-B': 'Berkshire Hathaway - Investment holding company',
                'JPM': 'JPMorgan Chase & Co. - Banking and financial services',
                'JNJ': 'Johnson & Johnson - Healthcare and pharmaceuticals',
                'V': 'Visa Inc. - Payment processing services',
                'PG': 'Procter & Gamble - Consumer goods',
                'UNH': 'UnitedHealth Group - Healthcare insurance',
                'HD': 'The Home Depot - Home improvement retail',
                'MA': 'Mastercard Inc. - Payment processing services',
                'DIS': 'The Walt Disney Company - Entertainment and media',
                'PYPL': 'PayPal Holdings - Digital payments',
                'ADBE': 'Adobe Inc. - Creative software and marketing',
                'CRM': 'Salesforce Inc. - Customer relationship management'
            }
            
            # Enhance stocks with descriptions
            enhanced_stocks = []
            for stock in result['stocks']:
                enhanced_stock = {
                    'symbol': stock['symbol'],
                    'description': popular_descriptions.get(stock['symbol'], f"{stock['symbol']} Corporation"),
                    'price': stock['price'],
                    'change': stock['change'],
                    'percent_change': stock['percent_change'],
                    'high': stock['high'],
                    'low': stock['low']
                }
                enhanced_stocks.append(enhanced_stock)
            
            return {
                'status': 'success',
                'stocks': enhanced_stocks
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching popular stocks: {str(e)}")

@app.get("/stocks/quotes")
async def get_stock_quotes(symbols: str):
    """Get real-time quotes for multiple stocks (comma-separated symbols)"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        result = await finnhub_service.get_multiple_quotes(symbol_list)
        if result['status'] == 'success':
            quotes = []
            for symbol, quote_data in result['quotes'].items():
                if quote_data['status'] == 'success':
                    quotes.append({
                        'symbol': quote_data['symbol'],
                        'current_price': quote_data['current_price'],
                        'change': quote_data['change'],
                        'percent_change': quote_data['percent_change'],
                        'high_price': quote_data['high_price'],
                        'low_price': quote_data['low_price'],
                        'open_price': quote_data['open_price'],
                        'previous_close': quote_data['previous_close'],
                        'timestamp': quote_data['timestamp']
                    })
            return {'status': 'success', 'quotes': quotes}
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quotes: {str(e)}")

@app.get("/stocks/quote/{symbol}")
async def get_stock_quote(symbol: str):
    """Get real-time quote for a single stock"""
    try:
        result = await finnhub_service.get_stock_quote(symbol.upper())
        if result['status'] == 'success':
            return {
                'status': 'success',
                'symbol': result['symbol'],
                'current_price': result['current_price'],
                'change': result['change'],
                'percent_change': result['percent_change'],
                'high_price': result['high_price'],
                'low_price': result['low_price'],
                'open_price': result['open_price'],
                'previous_close': result['previous_close'],
                'timestamp': result['timestamp']
            }
        else:
            raise HTTPException(status_code=404, detail=result.get('error', f'Quote not found for {symbol}'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quote for {symbol}: {str(e)}")

@app.get("/buffett-picks/quotes")
async def get_buffett_picks_quotes():
    """Get real-time quotes for Warren Buffett's top picks"""
    try:
        result = await finnhub_service.get_multiple_quotes(BUFFETT_TOP_PICKS)
        if result['status'] == 'success':
            picks_with_quotes = []
            
            # Company names for Buffett's picks
            company_names = {
                'AAPL': 'Apple Inc.',
                'BRK-B': 'Berkshire Hathaway Inc.',
                'KO': 'The Coca-Cola Company',
                'BAC': 'Bank of America Corporation',
                'CVX': 'Chevron Corporation',
                'OXY': 'Occidental Petroleum Corporation',
                'MCO': "Moody's Corporation",
                'AXP': 'American Express Company',
                'KHC': 'The Kraft Heinz Company',
                'VZ': 'Verizon Communications Inc.'
            }
            
            for symbol in BUFFETT_TOP_PICKS:
                quote_data = result['quotes'].get(symbol)
                if quote_data and quote_data['status'] == 'success':
                    picks_with_quotes.append({
                        'symbol': symbol,
                        'name': company_names.get(symbol, f"{symbol} Corporation"),
                        'price': quote_data['current_price'],
                        'change': quote_data['change'],
                        'change_percent': quote_data['percent_change']
                    })
            
            return {'status': 'success', 'stocks': picks_with_quotes}
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Buffett picks quotes: {str(e)}")

@app.get("/search/{query}")
async def search_stocks(query: str):
    """Search for stocks by symbol or company name using Finnhub"""
    try:
        # Use Finnhub search first
        finnhub_results = await finnhub_service.search_stocks(query)
        
        if finnhub_results:
            # Get quotes for the search results
            symbols = [stock.symbol for stock in finnhub_results[:10]]  # Limit to 10 results
            quotes = await finnhub_service.get_multiple_quotes(symbols)
            
            # Combine stock info with quotes
            results = []
            quote_dict = {q.symbol: q for q in quotes}
            
            for stock in finnhub_results[:10]:
                quote = quote_dict.get(stock.symbol)
                results.append({
                    "symbol": stock.symbol,
                    "name": stock.description,
                    "price": quote.current_price if quote else 0,
                    "change": quote.change if quote else 0,
                    "change_percent": quote.percent_change if quote else 0,
                    "market_cap": None  # Not available in Finnhub basic plan
                })
            
            return results
        
        # Fallback to yfinance if Finnhub doesn't return results
        ticker = yf.Ticker(query.upper())
        info = ticker.info
        
        if info.get('longName'):
            hist = ticker.history(period="1d")
            current_price = hist['Close'].iloc[-1] if len(hist) > 0 else info.get('currentPrice', 0)
            
            return [{
                "symbol": query.upper(),
                "name": info.get('longName', query),
                "price": round(current_price, 2),
                "market_cap": info.get('marketCap')
            }]
        else:
            return []
            
    except Exception as e:
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)