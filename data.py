"""Data module for fetching fundamental and price data from free sources.

This module provides a unified interface for fetching stock data from multiple
free data sources including Alpha Vantage, Yahoo Finance, and Finnhub.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class DataProvider:
    """Unified data provider for fundamental and price data."""
    
    def __init__(self, config: Dict):
        """Initialize data provider with configuration.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.primary_source = config.get('data_sources', {}).get('primary', 'yahoo')
        self.backup_source = config.get('data_sources', {}).get('backup', 'yahoo')
        
        # Initialize API clients
        self._init_alpha_vantage()
        self._init_finnhub()
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.2  # 200ms between requests
        
    def _init_alpha_vantage(self):
        """Initialize Alpha Vantage client."""
        api_key = self.config.get('data_sources', {}).get('api_keys', {}).get('alpha_vantage')
        if api_key and api_key != 'YOUR_API_KEY_HERE':
            self.av_fundamental = FundamentalData(key=api_key, output_format='pandas')
            self.av_timeseries = TimeSeries(key=api_key, output_format='pandas')
        else:
            self.av_fundamental = None
            self.av_timeseries = None
            
    def _init_finnhub(self):
        """Initialize Finnhub client."""
        self.finnhub_api_key = self.config.get('data_sources', {}).get('api_keys', {}).get('finnhub')
        if not self.finnhub_api_key or self.finnhub_api_key == 'YOUR_API_KEY_HERE':
            self.finnhub_api_key = None
            
    def _rate_limit(self, source: str):
        """Implement rate limiting for API calls."""
        now = time.time()
        if source in self.last_request_time:
            elapsed = now - self.last_request_time[source]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time[source] = time.time()
        
    def get_stock_universe(self) -> List[str]:
        """Get list of all NYSE/NASDAQ/AMEX stocks.
        
        Returns:
            List of stock symbols
        """
        try:
            # Use yfinance to get major exchange stocks
            # This is a simplified approach - in production, you might want
            # to use a more comprehensive source
            
            # Get S&P 500 stocks as a starting universe
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(sp500_url)
            sp500_symbols = tables[0]['Symbol'].tolist()
            
            # Add some major NASDAQ stocks
            nasdaq_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
            
            # Combine and deduplicate
            all_symbols = list(set(sp500_symbols + nasdaq_symbols))
            
            logger.info(f"Retrieved {len(all_symbols)} stocks from universe")
            return all_symbols
            
        except Exception as e:
            logger.error(f"Error fetching stock universe: {e}")
            # Fallback to a smaller list
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'VTI']
            
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing fundamental metrics or None if failed
        """
        try:
            # Try primary source first
            if self.primary_source == 'alpha_vantage':
                return self._get_fundamentals_alpha_vantage(symbol)
            elif self.primary_source == 'finnhub':
                return self._get_fundamentals_finnhub(symbol)
            else:
                return self._get_fundamentals_yahoo(symbol)
                
        except Exception as e:
            logger.warning(f"Primary source failed for {symbol}: {e}")
            
            # Try backup source
            try:
                if self.backup_source == 'yahoo':
                    return self._get_fundamentals_yahoo(symbol)
                elif self.backup_source == 'alpha_vantage':
                    return self._get_fundamentals_alpha_vantage(symbol)
                else:
                    return self._get_fundamentals_finnhub(symbol)
            except Exception as e2:
                logger.error(f"Backup source also failed for {symbol}: {e2}")
                return None
                
    def _get_fundamentals_yahoo(self, symbol: str) -> Optional[Dict]:
        """Get fundamentals from Yahoo Finance."""
        self._rate_limit('yahoo')
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            # Extract key metrics
            fundamentals = {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'trailing_pe': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'gross_margins': info.get('grossMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'profit_margins': info.get('profitMargins', 0),
                'free_cashflow': info.get('freeCashflow', 0),
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0),
                'current_ratio': info.get('currentRatio', 0),
                'book_value': info.get('bookValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
            }
            
            # Calculate additional metrics from financial statements
            if not financials.empty:
                try:
                    # Get TTM net income
                    if 'Net Income' in financials.index:
                        fundamentals['ttm_net_income'] = financials.loc['Net Income'].iloc[0]
                    
                    # Calculate revenue CAGR if we have enough data
                    if 'Total Revenue' in financials.index and len(financials.columns) >= 4:
                        revenues = financials.loc['Total Revenue'].dropna()
                        if len(revenues) >= 4:
                            fundamentals['revenue_cagr_5yr'] = self._calculate_cagr(
                                revenues.iloc[-1], revenues.iloc[0], len(revenues) - 1
                            )
                except Exception as e:
                    logger.warning(f"Error calculating additional metrics for {symbol}: {e}")
                    
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo fundamentals for {symbol}: {e}")
            return None
            
    def _get_fundamentals_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """Get fundamentals from Alpha Vantage."""
        if not self.av_fundamental:
            return None
            
        self._rate_limit('alpha_vantage')
        
        try:
            # Get company overview
            overview, _ = self.av_fundamental.get_company_overview(symbol)
            
            if overview.empty:
                return None
                
            # Get income statement
            income_statement, _ = self.av_fundamental.get_income_statement_annual(symbol)
            
            # Get balance sheet
            balance_sheet, _ = self.av_fundamental.get_balance_sheet_annual(symbol)
            
            # Get cash flow
            cash_flow, _ = self.av_fundamental.get_cash_flow_annual(symbol)
            
            # Extract metrics
            fundamentals = {
                'symbol': symbol,
                'market_cap': float(overview.get('MarketCapitalization', 0)),
                'trailing_pe': float(overview.get('TrailingPE', 0)),
                'forward_pe': float(overview.get('ForwardPE', 0)),
                'peg_ratio': float(overview.get('PEGRatio', 0)),
                'price_to_book': float(overview.get('PriceToBookRatio', 0)),
                'debt_to_equity': float(overview.get('DebtToEquityRatio', 0)),
                'return_on_equity': float(overview.get('ReturnOnEquityTTM', 0)),
                'revenue_growth': float(overview.get('QuarterlyRevenueGrowthYOY', 0)),
                'gross_margins': float(overview.get('GrossProfitTTM', 0)) / float(overview.get('RevenueTTM', 1)),
                'profit_margins': float(overview.get('ProfitMargin', 0)),
                'sector': overview.get('Sector', ''),
                'industry': overview.get('Industry', ''),
            }
            
            # Add income statement data
            if not income_statement.empty:
                fundamentals['ttm_net_income'] = float(income_statement.iloc[0].get('netIncome', 0))
                
                # Calculate revenue CAGR
                if len(income_statement) >= 5:
                    revenues = income_statement['totalRevenue'].astype(float)
                    fundamentals['revenue_cagr_5yr'] = self._calculate_cagr(
                        revenues.iloc[0], revenues.iloc[4], 4
                    )
                    
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage fundamentals for {symbol}: {e}")
            return None
            
    def _get_fundamentals_finnhub(self, symbol: str) -> Optional[Dict]:
        """Get fundamentals from Finnhub."""
        if not self.finnhub_api_key:
            return None
            
        self._rate_limit('finnhub')
        
        try:
            base_url = 'https://finnhub.io/api/v1'
            
            # Get basic financials
            url = f"{base_url}/stock/metric"
            params = {'symbol': symbol, 'metric': 'all', 'token': self.finnhub_api_key}
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            metric = data.get('metric', {})
            
            fundamentals = {
                'symbol': symbol,
                'market_cap': metric.get('marketCapitalization', 0),
                'trailing_pe': metric.get('peBasicExclExtraTTM', 0),
                'peg_ratio': metric.get('pegRatio', 0),
                'price_to_book': metric.get('pbAnnual', 0),
                'return_on_equity': metric.get('roeRfy', 0),
                'debt_to_equity': metric.get('totalDebt/totalEquityAnnual', 0),
                'gross_margins': metric.get('grossMarginAnnual', 0),
                'operating_margins': metric.get('operatingMarginAnnual', 0),
                'profit_margins': metric.get('netProfitMarginAnnual', 0),
                'current_ratio': metric.get('currentRatioAnnual', 0),
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub fundamentals for {symbol}: {e}")
            return None
            
    def get_price_data(self, symbol: str, period: str = '5y') -> Optional[pd.DataFrame]:
        """Get historical price data.
        
        Args:
            symbol: Stock symbol
            period: Time period (1y, 2y, 5y, 10y, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self._rate_limit('yahoo')
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
            
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if failed
        """
        try:
            self._rate_limit('yahoo')
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            
            if data.empty:
                return None
                
            return float(data['Close'].iloc[-1])
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
            
    def get_industry_peers(self, symbol: str) -> List[str]:
        """Get industry peer symbols.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of peer symbols
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            industry = info.get('industry', '')
            sector = info.get('sector', '')
            
            # This is a simplified approach - in production, you'd want
            # a more sophisticated industry classification
            universe = self.get_stock_universe()
            peers = []
            
            for peer_symbol in universe[:50]:  # Limit to avoid too many API calls
                if peer_symbol == symbol:
                    continue
                    
                try:
                    peer_ticker = yf.Ticker(peer_symbol)
                    peer_info = peer_ticker.info
                    
                    if (peer_info.get('industry', '') == industry or 
                        peer_info.get('sector', '') == sector):
                        peers.append(peer_symbol)
                        
                    if len(peers) >= 10:  # Limit number of peers
                        break
                        
                except Exception:
                    continue
                    
            return peers
            
        except Exception as e:
            logger.error(f"Error fetching industry peers for {symbol}: {e}")
            return []
            
    def _calculate_cagr(self, end_value: float, start_value: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate.
        
        Args:
            end_value: Ending value
            start_value: Starting value
            years: Number of years
            
        Returns:
            CAGR as decimal
        """
        if start_value <= 0 or years <= 0:
            return 0.0
            
        try:
            return (end_value / start_value) ** (1 / years) - 1
        except (ZeroDivisionError, ValueError):
            return 0.0
            
    def calculate_financial_metrics(self, fundamentals: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate additional financial metrics.
        
        Args:
            fundamentals: Basic fundamental data
            price_data: Historical price data
            
        Returns:
            Enhanced fundamentals with calculated metrics
        """
        enhanced = fundamentals.copy()
        
        try:
            # Calculate volatility
            if not price_data.empty and len(price_data) > 20:
                returns = price_data['Close'].pct_change().dropna()
                enhanced['volatility_1yr'] = returns.std() * np.sqrt(252)
                
            # Calculate free cash flow margin
            if 'free_cashflow' in fundamentals and 'market_cap' in fundamentals:
                if fundamentals['market_cap'] > 0:
                    # Estimate revenue from market cap and margins
                    revenue_estimate = fundamentals['market_cap'] * fundamentals.get('profit_margins', 0.1)
                    if revenue_estimate > 0:
                        enhanced['fcf_margin'] = fundamentals['free_cashflow'] / revenue_estimate
                        
            # Calculate debt-to-equity if not available
            if 'debt_to_equity' not in enhanced or enhanced['debt_to_equity'] == 0:
                if 'total_debt' in fundamentals and 'market_cap' in fundamentals:
                    if fundamentals['market_cap'] > 0:
                        enhanced['debt_to_equity'] = fundamentals['total_debt'] / fundamentals['market_cap']
                        
        except Exception as e:
            logger.warning(f"Error calculating enhanced metrics: {e}")
            
        return enhanced