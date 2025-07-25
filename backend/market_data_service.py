import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import json
from dataclasses import dataclass

@dataclass
class StockQuote:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None

@dataclass
class NewsItem:
    title: str
    summary: str
    url: str
    published_date: str
    source: str
    related_symbols: List[str]

class MarketDataService:
    """
    Service for fetching real-time market data, news, and analytics
    """
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 900  # 15 minutes cache for better rate limiting protection
        self.last_api_call = {}  # Track last API call per symbol
        self.min_call_interval = 2  # Minimum 2 seconds between calls per symbol
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        cached_time = self.cache[key].get('timestamp')
        if not cached_time:
            return False
        
        return datetime.now() - cached_time < timedelta(seconds=self.cache_duration)
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data if valid"""
        if self._is_cache_valid(key):
            return self.cache[key]['data']
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Set data in cache"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock quote with aggressive caching and rate limiting protection"""
        cache_key = f"quote_{symbol}"
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data:
            return cached_data
        
        # Check rate limiting
        now = datetime.now()
        if symbol in self.last_api_call:
            time_since_last = (now - self.last_api_call[symbol]).total_seconds()
            if time_since_last < self.min_call_interval:
                # Return cached data even if expired, or error
                if cache_key in self.cache:
                    return self.cache[cache_key]['data']
                return {'status': 'error', 'error': f'Rate limited for {symbol}'}
        
        try:
            # Update last call time
            self.last_api_call[symbol] = now
            
            # Use yfinance for stock data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return {'status': 'error', 'error': f'No data found for symbol {symbol}'}
            
            current_price = hist['Close'].iloc[-1]
            previous_close = info.get('previousClose', current_price)
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close else 0
            
            quote_data = {
                'status': 'success',
                'symbol': symbol.upper(),
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'timestamp': datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, quote_data)
            return quote_data
            
        except Exception as e:
            return {'status': 'error', 'error': f'Failed to get quote for {symbol}: {str(e)}'}
    
    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get quotes for multiple symbols with aggressive rate limiting protection"""
        try:
            quotes = {}
            
            # Process symbols in smaller batches with delays to avoid overwhelming the API
            batch_size = 3  # Reduced batch size
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                # Add delay between batches
                if i > 0:
                    await asyncio.sleep(3)  # 3 second delay between batches
                
                # Process batch sequentially to avoid rate limiting
                for symbol in batch:
                    result = await self.get_stock_quote(symbol)
                    quotes[symbol] = result
                    
                    # Small delay between individual calls
                    await asyncio.sleep(1)
            
            return {'status': 'success', 'quotes': quotes}
            
        except Exception as e:
            return {'status': 'error', 'error': f'Failed to get multiple quotes: {str(e)}'}
    
    async def search_stocks(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for stocks by name or symbol"""
        try:
            # For demo purposes, we'll use a simple search approach
            # In production, you might want to use a dedicated search API
            
            # Try to get ticker info if query looks like a symbol
            if len(query) <= 5 and query.isalpha():
                try:
                    ticker = yf.Ticker(query.upper())
                    info = ticker.info
                    
                    if info and info.get('longName'):
                        return {
                            'status': 'success',
                            'results': [{
                                'symbol': query.upper(),
                                'name': info.get('longName'),
                                'sector': info.get('sector'),
                                'industry': info.get('industry'),
                                'market_cap': info.get('marketCap'),
                                'exchange': info.get('exchange')
                            }]
                        }
                except:
                    pass
            
            # For more comprehensive search, you would integrate with:
            # - Yahoo Finance search API
            # - Alpha Vantage symbol search
            # - Financial Modeling Prep search
            # - Or maintain your own symbol database
            
            # Demo response
            demo_results = [
                {
                    'symbol': 'AAPL',
                    'name': 'Apple Inc.',
                    'sector': 'Technology',
                    'industry': 'Consumer Electronics',
                    'exchange': 'NASDAQ'
                },
                {
                    'symbol': 'MSFT',
                    'name': 'Microsoft Corporation',
                    'sector': 'Technology',
                    'industry': 'Software',
                    'exchange': 'NASDAQ'
                }
            ] if query.lower() in ['apple', 'aapl', 'tech', 'technology'] else []
            
            return {'status': 'success', 'results': demo_results[:limit]}
            
        except Exception as e:
            return {'status': 'error', 'error': f'Search failed: {str(e)}'}
    
    async def get_market_movers(self, mover_type: str = "gainers") -> Dict[str, Any]:
        """Get market movers (gainers, losers, most active)"""
        cache_key = f"movers_{mover_type}"
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # For demo purposes, return sample data
            # In production, you would fetch from:
            # - Yahoo Finance screeners
            # - Financial APIs
            # - Market data providers
            
            sample_movers = {
                'gainers': [
                    {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'change_percent': 5.2, 'price': 485.50},
                    {'symbol': 'AMD', 'name': 'Advanced Micro Devices', 'change_percent': 4.8, 'price': 142.30},
                    {'symbol': 'TSLA', 'name': 'Tesla Inc', 'change_percent': 3.9, 'price': 248.75}
                ],
                'losers': [
                    {'symbol': 'META', 'name': 'Meta Platforms Inc', 'change_percent': -2.1, 'price': 325.80},
                    {'symbol': 'NFLX', 'name': 'Netflix Inc', 'change_percent': -1.8, 'price': 485.20},
                    {'symbol': 'GOOGL', 'name': 'Alphabet Inc', 'change_percent': -1.5, 'price': 138.45}
                ],
                'most_active': [
                    {'symbol': 'AAPL', 'name': 'Apple Inc', 'volume': 45000000, 'price': 189.50},
                    {'symbol': 'TSLA', 'name': 'Tesla Inc', 'volume': 38000000, 'price': 248.75},
                    {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF', 'volume': 35000000, 'price': 475.20}
                ]
            }
            
            result = {
                'status': 'success',
                'type': mover_type,
                'data': sample_movers.get(mover_type, []),
                'timestamp': datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            return {'status': 'error', 'error': f'Failed to get market movers: {str(e)}'}
    
    async def get_stock_news(self, symbol: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Get stock news (general market news if no symbol provided)"""
        cache_key = f"news_{symbol or 'general'}_{limit}"
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # For demo purposes, return sample news
            # In production, integrate with:
            # - Yahoo Finance news API
            # - Alpha Vantage news
            # - NewsAPI
            # - Financial news providers
            
            sample_news = [
                {
                    'title': 'Market Rally Continues as Tech Stocks Surge',
                    'summary': 'Technology stocks led the market higher today as investors showed renewed confidence in growth stocks.',
                    'url': 'https://example.com/news/1',
                    'published_date': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'source': 'Financial News',
                    'related_symbols': ['AAPL', 'MSFT', 'GOOGL'] if not symbol else [symbol]
                },
                {
                    'title': 'Federal Reserve Signals Potential Rate Changes',
                    'summary': 'The Federal Reserve indicated possible adjustments to interest rates in upcoming meetings.',
                    'url': 'https://example.com/news/2',
                    'published_date': (datetime.now() - timedelta(hours=4)).isoformat(),
                    'source': 'Economic Times',
                    'related_symbols': ['SPY', 'QQQ', 'IWM'] if not symbol else [symbol]
                },
                {
                    'title': 'Earnings Season Kicks Off with Strong Results',
                    'summary': 'Several major companies reported better-than-expected earnings, boosting investor sentiment.',
                    'url': 'https://example.com/news/3',
                    'published_date': (datetime.now() - timedelta(hours=6)).isoformat(),
                    'source': 'Market Watch',
                    'related_symbols': ['AAPL', 'MSFT', 'AMZN'] if not symbol else [symbol]
                }
            ]
            
            # Filter news by symbol if provided
            if symbol:
                filtered_news = [
                    news for news in sample_news 
                    if symbol.upper() in news['related_symbols']
                ]
                if not filtered_news:
                    # Add symbol-specific news
                    filtered_news = [{
                        'title': f'{symbol.upper()} Shows Strong Performance',
                        'summary': f'Latest analysis shows {symbol.upper()} demonstrating solid fundamentals and growth potential.',
                        'url': f'https://example.com/news/{symbol.lower()}',
                        'published_date': (datetime.now() - timedelta(hours=1)).isoformat(),
                        'source': 'Stock Analysis',
                        'related_symbols': [symbol.upper()]
                    }]
                sample_news = filtered_news
            
            result = {
                'status': 'success',
                'symbol': symbol,
                'news': sample_news[:limit],
                'timestamp': datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            return {'status': 'error', 'error': f'Failed to get news: {str(e)}'}
    
    async def get_stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock analysis"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return {'status': 'error', 'error': f'No data found for {symbol}'}
            
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            
            # Calculate volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
            
            # Price targets and recommendations
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_mean = info.get('targetMeanPrice')
            recommendation = info.get('recommendationKey', 'hold')
            
            analysis = {
                'status': 'success',
                'symbol': symbol.upper(),
                'current_price': round(current_price, 2),
                'technical_analysis': {
                    'sma_20': round(sma_20, 2) if pd.notna(sma_20) else None,
                    'sma_50': round(sma_50, 2) if pd.notna(sma_50) else None,
                    'volatility': round(volatility, 4),
                    'trend': 'bullish' if current_price > sma_20 > sma_50 else 'bearish'
                },
                'fundamental_analysis': {
                    'pe_ratio': info.get('trailingPE'),
                    'peg_ratio': info.get('pegRatio'),
                    'price_to_book': info.get('priceToBook'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'roe': info.get('returnOnEquity'),
                    'profit_margin': info.get('profitMargins')
                },
                'analyst_targets': {
                    'high': target_high,
                    'low': target_low,
                    'mean': target_mean,
                    'recommendation': recommendation
                },
                'risk_assessment': {
                    'risk_level': 'high' if volatility > 0.3 else 'medium' if volatility > 0.2 else 'low',
                    'beta': info.get('beta'),
                    'volatility': round(volatility, 4)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            return {'status': 'error', 'error': f'Analysis failed for {symbol}: {str(e)}'}
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview"""
        cache_key = "market_overview"
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            # Get major indices
            indices = ['SPY', 'QQQ', 'IWM', 'VTI']
            index_data = {}
            
            for index in indices:
                quote = await self.get_stock_quote(index)
                if quote['status'] == 'success':
                    index_data[index] = {
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent']
                    }
            
            # Market sentiment (simplified)
            positive_changes = sum(1 for data in index_data.values() if data['change'] > 0)
            total_indices = len(index_data)
            sentiment = 'bullish' if positive_changes > total_indices / 2 else 'bearish'
            
            overview = {
                'status': 'success',
                'indices': index_data,
                'market_sentiment': sentiment,
                'timestamp': datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, overview)
            return overview
            
        except Exception as e:
            return {'status': 'error', 'error': f'Failed to get market overview: {str(e)}'}