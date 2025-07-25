import finnhub
import asyncio
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinnhubStock:
    symbol: str
    description: str
    display_symbol: str
    type: str
    currency: str = "USD"
    mic: Optional[str] = None

@dataclass
class FinnhubQuote:
    symbol: str
    current_price: float
    change: float
    percent_change: float
    high_price: float
    low_price: float
    open_price: float
    previous_close: float
    timestamp: int

class FinnhubService:
    """
    Service for fetching data from Finnhub API
    Provides real-time stock data, quotes, and comprehensive stock listings
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("Finnhub API key is required. Set FINNHUB_API_KEY environment variable.")
        
        self.client = finnhub.Client(api_key=self.api_key)
        
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache for stock lists
        self.quote_cache_duration = 60  # 1 minute cache for quotes
        self.last_api_call = {}
        self.min_call_interval = 1.2  # 1.2 seconds between calls to respect rate limits
        self.max_retries = 3  # Maximum number of retries for failed requests
        
    def _is_cache_valid(self, key: str, duration: int = None) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        cached_time = self.cache[key].get('timestamp')
        if not cached_time:
            return False
        
        cache_duration = duration or self.cache_duration
        return datetime.now() - cached_time < timedelta(seconds=cache_duration)
    
    def _get_cached_data(self, key: str, duration: int = None) -> Optional[Any]:
        """Get cached data if valid"""
        if self._is_cache_valid(key, duration):
            return self.cache[key]['data']
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Set data in cache"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def _rate_limit_check(self, symbol: str = "general") -> bool:
        """Check if we can make an API call without hitting rate limits"""
        now = datetime.now()
        if symbol in self.last_api_call:
            time_since_last = (now - self.last_api_call[symbol]).total_seconds()
            if time_since_last < self.min_call_interval:
                return False
        
        self.last_api_call[symbol] = now
        return True
    
    async def get_all_us_stocks(self) -> Dict[str, Any]:
        """Get all US stock symbols from Finnhub"""
        cache_key = "us_stocks_all"
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data:
            logger.info("Returning cached US stocks data")
            return cached_data
        
        if not await self._rate_limit_check("stock_symbols"):
            logger.warning("Rate limit hit for stock symbols")
            if cache_key in self.cache:
                return self.cache[cache_key]['data']
            return {'status': 'error', 'error': 'Rate limited'}
        
        try:
            logger.info("Fetching US stocks from Finnhub API")
            stocks_data = self.client.stock_symbols('US')
            
            # Filter and format the data
            formatted_stocks = []
            for stock in stocks_data:
                if stock.get('type') == 'Common Stock' and stock.get('symbol'):
                    formatted_stock = FinnhubStock(
                        symbol=stock['symbol'],
                        description=stock.get('description', ''),
                        display_symbol=stock.get('displaySymbol', stock['symbol']),
                        type=stock.get('type', ''),
                        currency=stock.get('currency', 'USD'),
                        mic=stock.get('mic')
                    )
                    formatted_stocks.append(formatted_stock.__dict__)
            
            result = {
                'status': 'success',
                'count': len(formatted_stocks),
                'stocks': formatted_stocks,
                'timestamp': datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            logger.info(f"Successfully fetched {len(formatted_stocks)} US stocks")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching US stocks: {str(e)}")
            return {'status': 'error', 'error': f'Failed to fetch US stocks: {str(e)}'}
    
    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a specific stock with retry logic"""
        cache_key = f"quote_{symbol}"
        cached_data = self._get_cached_data(cache_key, self.quote_cache_duration)
        
        if cached_data:
            return cached_data
        
        if not await self._rate_limit_check(symbol):
            logger.warning(f"Rate limit hit for {symbol}")
            if cache_key in self.cache:
                return self.cache[cache_key]['data']
            return {'status': 'error', 'error': f'Rate limited for {symbol}'}
        
        # Retry logic for API calls
        for attempt in range(self.max_retries):
            try:
                quote_data = self.client.quote(symbol)
                
                if not quote_data or 'c' not in quote_data:
                    logger.warning(f"No quote data found for {symbol}")
                    return {'status': 'error', 'error': f'No quote data found for {symbol}'}
                
                # Format the quote data
                formatted_quote = {
                    'status': 'success',
                    'symbol': symbol.upper(),
                    'current_price': quote_data.get('c', 0),  # Current price
                    'change': quote_data.get('d', 0),  # Change
                    'percent_change': quote_data.get('dp', 0),  # Percent change
                    'high_price': quote_data.get('h', 0),  # High price of the day
                    'low_price': quote_data.get('l', 0),  # Low price of the day
                    'open_price': quote_data.get('o', 0),  # Open price of the day
                    'previous_close': quote_data.get('pc', 0),  # Previous close price
                    'timestamp': quote_data.get('t', int(datetime.now().timestamp())),
                    'updated_at': datetime.now().isoformat()
                }
                
                self._set_cache(cache_key, formatted_quote)
                return formatted_quote
                

            
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout for {symbol} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {'status': 'error', 'error': f'Request timeout for {symbol} after {self.max_retries} attempts'}
            
            except Exception as e:
                logger.error(f"Error fetching quote for {symbol} (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {'status': 'error', 'error': f'Failed to get quote for {symbol}: {str(e)}'}
    
    async def get_multiple_quotes(self, symbols: List[str], batch_size: int = 3) -> Dict[str, Any]:
        """Get quotes for multiple symbols with rate limiting and error handling"""
        try:
            quotes = {}
            
            # Process symbols in smaller batches to respect rate limits
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                # Add longer delay between batches
                if i > 0:
                    try:
                        await asyncio.sleep(3)
                    except asyncio.CancelledError:
                        logger.warning("Request was cancelled during batch sleep.")
                        return {'status': 'error', 'error': 'Request was cancelled by the client.'}
                
                # Process batch
                for symbol in batch:
                    try:
                        result = await self.get_stock_quote(symbol)
                        quotes[symbol] = result
                        # Longer delay between individual calls
                        await asyncio.sleep(1.5)
                    except asyncio.CancelledError:
                        logger.warning(f"Request was cancelled during quote fetch for {symbol}.")
                        return {'status': 'error', 'error': 'Request was cancelled by the client.'}
            
            return {'status': 'success', 'quotes': quotes}
            
        except Exception as e:
            logger.error(f"Error fetching multiple quotes: {str(e)}")
            return {'status': 'error', 'error': f'Failed to get multiple quotes: {str(e)}'}
    
    async def search_stocks(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Search stocks by symbol or description"""
        try:
            # Get all stocks first
            all_stocks_result = await self.get_all_us_stocks()
            
            if all_stocks_result['status'] != 'success':
                return all_stocks_result
            
            stocks = all_stocks_result['stocks']
            query_upper = query.upper()
            
            # Filter stocks based on query
            matching_stocks = []
            for stock in stocks:
                if (query_upper in stock['symbol'].upper() or 
                    query_upper in stock['description'].upper()):
                    matching_stocks.append(stock)
                
                if len(matching_stocks) >= limit:
                    break
            
            return {
                'status': 'success',
                'query': query,
                'count': len(matching_stocks),
                'results': matching_stocks
            }
            
        except Exception as e:
            logger.error(f"Error searching stocks: {str(e)}")
            return {'status': 'error', 'error': f'Failed to search stocks: {str(e)}'}
    
    async def get_popular_stocks(self, limit: int = 50) -> Dict[str, Any]:
        """Get popular/major stocks with their quotes"""
        popular_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL',
            'ADBE', 'CRM', 'INTC', 'VZ', 'KO', 'PFE', 'T', 'CSCO', 'XOM', 'ABT',
            'TMO', 'CVX', 'ACN', 'COST', 'AVGO', 'DHR', 'TXN', 'NEE', 'LLY',
            'WMT', 'ORCL', 'MDT', 'QCOM', 'HON', 'UNP', 'LOW', 'IBM', 'SBUX',
            'AMD', 'LIN', 'CAT'
        ]
        
        # Limit to requested number
        symbols_to_fetch = popular_symbols[:limit]
        
        try:
            quotes_result = await self.get_multiple_quotes(symbols_to_fetch, batch_size=2)
            
            if quotes_result['status'] != 'success':
                return quotes_result
            
            # Format the response
            popular_stocks = []
            for symbol, quote_data in quotes_result['quotes'].items():
                if quote_data['status'] == 'success':
                    popular_stocks.append({
                        'symbol': symbol,
                        'price': quote_data['current_price'],
                        'change': quote_data['change'],
                        'percent_change': quote_data['percent_change'],
                        'high': quote_data['high_price'],
                        'low': quote_data['low_price'],
                        'volume': 0  # Finnhub quote doesn't include volume in basic plan
                    })
            
            return {
                'status': 'success',
                'count': len(popular_stocks),
                'stocks': popular_stocks,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching popular stocks: {str(e)}")
            return {'status': 'error', 'error': f'Failed to get popular stocks: {str(e)}'}