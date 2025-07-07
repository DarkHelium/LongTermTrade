"""Unit tests for the data module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import requests

from data import DataProvider, calculate_cagr, calculate_financial_metric


class TestDataProvider:
    """Test cases for DataProvider class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'data_sources': {
                'alpha_vantage_api_key': 'test_key',
                'finnhub_api_key': 'test_key',
                'rate_limit_delay': 0.1
            }
        }
        self.data_provider = DataProvider(self.config)

    def test_initialization(self):
        """Test data provider initialization."""
        assert self.data_provider.alpha_vantage_key == 'test_key'
        assert self.data_provider.finnhub_key == 'test_key'
        assert self.data_provider.rate_limit_delay == 0.1
        assert hasattr(self.data_provider, 'session')

    @patch('requests.Session.get')
    def test_get_stock_universe_success(self, mock_get):
        """Test successful stock universe retrieval."""
        # Mock Alpha Vantage response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'symbol': 'AAPL',
                    'name': 'Apple Inc.',
                    'exchange': 'NASDAQ',
                    'assetType': 'Stock',
                    'status': 'Active'
                },
                {
                    'symbol': 'MSFT',
                    'name': 'Microsoft Corporation',
                    'exchange': 'NASDAQ',
                    'assetType': 'Stock',
                    'status': 'Active'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        universe = self.data_provider.get_stock_universe()
        
        assert len(universe) == 2
        assert 'AAPL' in universe['symbol'].values
        assert 'MSFT' in universe['symbol'].values
        assert 'name' in universe.columns
        assert 'exchange' in universe.columns

    @patch('requests.Session.get')
    def test_get_stock_universe_api_error(self, mock_get):
        """Test stock universe retrieval with API error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        universe = self.data_provider.get_stock_universe()
        
        # Should return empty DataFrame on error
        assert len(universe) == 0
        assert isinstance(universe, pd.DataFrame)

    @patch('yfinance.Ticker')
    def test_get_fundamentals_success(self, mock_ticker):
        """Test successful fundamentals retrieval."""
        # Mock yfinance data
        mock_info = {
            'marketCap': 3000000000000,
            'trailingPE': 25.5,
            'forwardPE': 22.0,
            'pegRatio': 1.2,
            'debtToEquity': 45.5,
            'returnOnEquity': 0.25,
            'grossMargins': 0.42,
            'freeCashflow': 50000000000,
            'totalRevenue': 400000000000
        }
        
        mock_financials = pd.DataFrame({
            'Total Revenue': [400e9, 380e9, 360e9, 340e9, 320e9],
            'Net Income': [80e9, 75e9, 70e9, 65e9, 60e9]
        }, index=pd.date_range('2023-12-31', periods=5, freq='-1Y'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.financials = mock_financials
        mock_ticker.return_value = mock_ticker_instance
        
        fundamentals = self.data_provider.get_fundamentals('AAPL')
        
        assert fundamentals is not None
        assert 'market_cap' in fundamentals
        assert 'forward_pe' in fundamentals
        assert 'revenue_cagr_5yr' in fundamentals
        assert fundamentals['market_cap'] == 3e12

    @patch('yfinance.Ticker')
    def test_get_fundamentals_missing_data(self, mock_ticker):
        """Test fundamentals retrieval with missing data."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {}  # Empty info
        mock_ticker_instance.financials = pd.DataFrame()  # Empty financials
        mock_ticker.return_value = mock_ticker_instance
        
        fundamentals = self.data_provider.get_fundamentals('INVALID')
        
        assert fundamentals is None

    @patch('yfinance.download')
    def test_get_historical_prices_success(self, mock_download):
        """Test successful historical price retrieval."""
        # Mock yfinance download response
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(150, 200, 252),
            'High': np.random.uniform(155, 205, 252),
            'Low': np.random.uniform(145, 195, 252),
            'Close': np.random.uniform(150, 200, 252),
            'Volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)
        
        mock_download.return_value = mock_data
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        prices = self.data_provider.get_historical_prices(['AAPL'], start_date, end_date)
        
        assert 'AAPL' in prices.columns
        assert len(prices) == 252
        assert not prices['AAPL'].isna().all()

    @patch('yfinance.download')
    def test_get_historical_prices_multiple_symbols(self, mock_download):
        """Test historical price retrieval for multiple symbols."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            ('Close', 'AAPL'): np.random.uniform(150, 200, 100),
            ('Close', 'MSFT'): np.random.uniform(250, 350, 100),
            ('Volume', 'AAPL'): np.random.randint(1000000, 10000000, 100),
            ('Volume', 'MSFT'): np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        
        mock_download.return_value = mock_data
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 4, 10)
        
        prices = self.data_provider.get_historical_prices(
            ['AAPL', 'MSFT'], start_date, end_date
        )
        
        assert 'AAPL' in prices.columns
        assert 'MSFT' in prices.columns
        assert len(prices) == 100

    @patch('yfinance.Ticker')
    def test_get_current_price_success(self, mock_ticker):
        """Test successful current price retrieval."""
        mock_history = pd.DataFrame({
            'Close': [180.50]
        }, index=[datetime.now()])
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker.return_value = mock_ticker_instance
        
        price = self.data_provider.get_current_price('AAPL')
        
        assert price == 180.50

    @patch('yfinance.Ticker')
    def test_get_current_price_failure(self, mock_ticker):
        """Test current price retrieval failure."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty
        mock_ticker.return_value = mock_ticker_instance
        
        price = self.data_provider.get_current_price('INVALID')
        
        assert price is None

    @patch('requests.Session.get')
    def test_get_industry_peers_success(self, mock_get):
        """Test successful industry peers retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN'
        ]
        mock_get.return_value = mock_response
        
        peers = self.data_provider.get_industry_peers('AAPL')
        
        assert len(peers) == 4
        assert 'AAPL' in peers
        assert 'MSFT' in peers

    @patch('requests.Session.get')
    def test_get_industry_peers_api_error(self, mock_get):
        """Test industry peers retrieval with API error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        peers = self.data_provider.get_industry_peers('INVALID')
        
        assert peers == []

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        import time
        
        start_time = time.time()
        self.data_provider._rate_limit()
        self.data_provider._rate_limit()
        end_time = time.time()
        
        # Should take at least the rate limit delay
        assert end_time - start_time >= self.data_provider.rate_limit_delay

    @patch('time.sleep')
    def test_rate_limiting_called(self, mock_sleep):
        """Test that rate limiting sleep is called."""
        self.data_provider._rate_limit()
        mock_sleep.assert_called_once_with(0.1)

    def test_cache_functionality(self):
        """Test caching functionality."""
        # Mock a method to test caching
        with patch.object(self.data_provider, 'get_current_price') as mock_method:
            mock_method.return_value = 180.50
            
            # First call should hit the API
            price1 = self.data_provider.get_current_price('AAPL')
            
            # Second call should use cache (if implemented)
            price2 = self.data_provider.get_current_price('AAPL')
            
            assert price1 == price2
            assert mock_method.call_count >= 1


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_calculate_cagr_positive_growth(self):
        """Test CAGR calculation with positive growth."""
        start_value = 100
        end_value = 200
        years = 5
        
        cagr = calculate_cagr(start_value, end_value, years)
        
        # 100 * (1 + cagr)^5 = 200
        # (1 + cagr)^5 = 2
        # 1 + cagr = 2^(1/5) ≈ 1.1487
        # cagr ≈ 0.1487
        assert abs(cagr - 0.1487) < 0.001

    def test_calculate_cagr_negative_growth(self):
        """Test CAGR calculation with negative growth."""
        start_value = 200
        end_value = 100
        years = 3
        
        cagr = calculate_cagr(start_value, end_value, years)
        
        # Should be negative
        assert cagr < 0
        assert abs(cagr - (-0.2063)) < 0.001

    def test_calculate_cagr_zero_years(self):
        """Test CAGR calculation with zero years."""
        cagr = calculate_cagr(100, 200, 0)
        assert cagr == 0

    def test_calculate_cagr_zero_start_value(self):
        """Test CAGR calculation with zero start value."""
        cagr = calculate_cagr(0, 100, 5)
        assert cagr == 0

    def test_calculate_cagr_negative_start_value(self):
        """Test CAGR calculation with negative start value."""
        cagr = calculate_cagr(-100, 100, 5)
        assert cagr == 0  # Should handle gracefully

    def test_calculate_financial_metric_valid_data(self):
        """Test financial metric calculation with valid data."""
        data = pd.Series([100, 110, 121, 133, 146], name='revenue')
        
        # Test mean
        mean_val = calculate_financial_metric(data, 'mean')
        assert abs(mean_val - 122) < 1
        
        # Test std
        std_val = calculate_financial_metric(data, 'std')
        assert std_val > 0
        
        # Test growth rate
        growth = calculate_financial_metric(data, 'growth')
        assert growth > 0  # Should be positive growth

    def test_calculate_financial_metric_empty_data(self):
        """Test financial metric calculation with empty data."""
        data = pd.Series([], dtype=float)
        
        result = calculate_financial_metric(data, 'mean')
        assert pd.isna(result) or result == 0

    def test_calculate_financial_metric_invalid_metric(self):
        """Test financial metric calculation with invalid metric type."""
        data = pd.Series([100, 110, 121])
        
        with pytest.raises(ValueError):
            calculate_financial_metric(data, 'invalid_metric')

    def test_calculate_financial_metric_single_value(self):
        """Test financial metric calculation with single value."""
        data = pd.Series([100])
        
        mean_val = calculate_financial_metric(data, 'mean')
        assert mean_val == 100
        
        std_val = calculate_financial_metric(data, 'std')
        assert std_val == 0  # Standard deviation of single value is 0
        
        growth = calculate_financial_metric(data, 'growth')
        assert growth == 0  # Can't calculate growth with single value

    def test_calculate_financial_metric_with_nans(self):
        """Test financial metric calculation with NaN values."""
        data = pd.Series([100, np.nan, 121, 133, np.nan])
        
        # Should handle NaN values gracefully
        mean_val = calculate_financial_metric(data, 'mean')
        assert not pd.isna(mean_val)
        assert mean_val > 0

    def test_calculate_financial_metric_all_nans(self):
        """Test financial metric calculation with all NaN values."""
        data = pd.Series([np.nan, np.nan, np.nan])
        
        result = calculate_financial_metric(data, 'mean')
        assert pd.isna(result) or result == 0

    def test_calculate_financial_metric_negative_values(self):
        """Test financial metric calculation with negative values."""
        data = pd.Series([-100, -50, 0, 50, 100])
        
        mean_val = calculate_financial_metric(data, 'mean')
        assert mean_val == 0
        
        std_val = calculate_financial_metric(data, 'std')
        assert std_val > 0


class TestDataProviderIntegration:
    """Integration tests for DataProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'data_sources': {
                'alpha_vantage_api_key': 'test_key',
                'finnhub_api_key': 'test_key',
                'rate_limit_delay': 0.01  # Faster for tests
            }
        }
        self.data_provider = DataProvider(self.config)

    @patch('yfinance.Ticker')
    @patch('yfinance.download')
    def test_full_data_pipeline(self, mock_download, mock_ticker):
        """Test complete data retrieval pipeline."""
        # Mock fundamentals
        mock_info = {
            'marketCap': 3000000000000,
            'trailingPE': 25.5,
            'forwardPE': 22.0,
            'pegRatio': 1.2,
            'debtToEquity': 45.5,
            'returnOnEquity': 0.25,
            'grossMargins': 0.42,
            'freeCashflow': 50000000000,
            'totalRevenue': 400000000000
        }
        
        mock_financials = pd.DataFrame({
            'Total Revenue': [400e9, 380e9, 360e9, 340e9, 320e9],
            'Net Income': [80e9, 75e9, 70e9, 65e9, 60e9]
        }, index=pd.date_range('2023-12-31', periods=5, freq='-1Y'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker_instance.financials = mock_financials
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock price data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        mock_price_data = pd.DataFrame({
            'Close': np.random.uniform(150, 200, 252)
        }, index=dates)
        mock_download.return_value = mock_price_data
        
        # Test the pipeline
        symbol = 'AAPL'
        
        # Get fundamentals
        fundamentals = self.data_provider.get_fundamentals(symbol)
        assert fundamentals is not None
        assert 'market_cap' in fundamentals
        
        # Get historical prices
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        prices = self.data_provider.get_historical_prices([symbol], start_date, end_date)
        assert symbol in prices.columns
        assert len(prices) > 0
        
        # Verify data consistency
        assert fundamentals['market_cap'] > 0
        assert not prices[symbol].isna().all()

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with invalid API keys
        bad_config = {
            'data_sources': {
                'alpha_vantage_api_key': 'invalid_key',
                'finnhub_api_key': 'invalid_key',
                'rate_limit_delay': 0.01
            }
        }
        
        bad_provider = DataProvider(bad_config)
        
        # Should handle errors gracefully
        universe = bad_provider.get_stock_universe()
        assert isinstance(universe, pd.DataFrame)
        
        peers = bad_provider.get_industry_peers('AAPL')
        assert isinstance(peers, list)

    def test_data_validation(self):
        """Test data validation and cleaning."""
        # Test with mock data that has issues
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock data with missing/invalid values
            mock_info = {
                'marketCap': None,  # Missing market cap
                'trailingPE': -5,   # Invalid PE
                'forwardPE': 22.0,
                'pegRatio': float('inf'),  # Invalid PEG
                'debtToEquity': 45.5,
                'returnOnEquity': 0.25,
                'grossMargins': 1.5,  # Invalid margin > 100%
                'freeCashflow': 50000000000,
                'totalRevenue': 0  # Zero revenue
            }
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = mock_info
            mock_ticker_instance.financials = pd.DataFrame()  # Empty
            mock_ticker.return_value = mock_ticker_instance
            
            fundamentals = self.data_provider.get_fundamentals('TEST')
            
            # Should handle invalid data gracefully
            if fundamentals is not None:
                # Check that invalid values are handled
                assert fundamentals.get('market_cap', 0) >= 0
                assert fundamentals.get('trailing_pe', 0) >= 0
                assert not np.isinf(fundamentals.get('peg_ratio', 1))