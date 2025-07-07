"""Unit tests for the portfolio module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from portfolio import PortfolioManager


class TestPortfolioManager:
    """Test cases for PortfolioManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'portfolio': {
                'core_allocation': 0.70,
                'satellite_allocation': 0.30,
                'core_etfs': {
                    'VTI': 0.60,
                    'VXUS': 0.40
                },
                'max_satellite_stocks': 20,
                'rebalance_threshold': 0.05,
                'min_trade_amount': 100
            },
            'risk_management': {
                'max_position_size': 0.20,
                'max_sector_allocation': 0.25,
                'cash_buffer': 0.02
            }
        }
        
        self.mock_broker = Mock()
        self.portfolio_manager = PortfolioManager(self.config, self.mock_broker)

    def test_initialization(self):
        """Test portfolio manager initialization."""
        assert self.portfolio_manager.core_allocation == 0.70
        assert self.portfolio_manager.satellite_allocation == 0.30
        assert self.portfolio_manager.core_etfs == {'VTI': 0.60, 'VXUS': 0.40}
        assert self.portfolio_manager.max_satellite_stocks == 20

    def test_calculate_target_allocation_empty_portfolio(self):
        """Test target allocation calculation with empty portfolio."""
        selected_stocks = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'composite_score': [3.5, 3.2, 3.8]
        })
        
        portfolio_value = 100000
        
        allocation = self.portfolio_manager.calculate_target_allocation(
            selected_stocks, portfolio_value
        )
        
        # Check core ETF allocations
        assert allocation['VTI'] == 42000  # 70% * 60% * 100k
        assert allocation['VXUS'] == 28000  # 70% * 40% * 100k
        
        # Check satellite allocations (30% split equally among 3 stocks)
        expected_per_stock = 30000 / 3  # 10k each
        assert allocation['AAPL'] == expected_per_stock
        assert allocation['MSFT'] == expected_per_stock
        assert allocation['GOOGL'] == expected_per_stock

    def test_calculate_target_allocation_with_existing_holdings(self):
        """Test target allocation with existing holdings."""
        selected_stocks = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'composite_score': [3.5, 3.2]
        })
        
        current_holdings = {
            'VTI': {'shares': 100, 'market_value': 25000},
            'AAPL': {'shares': 50, 'market_value': 8000},
            'NVDA': {'shares': 20, 'market_value': 12000}  # Not in selected stocks
        }
        
        portfolio_value = 100000
        
        allocation = self.portfolio_manager.calculate_target_allocation(
            selected_stocks, portfolio_value, current_holdings
        )
        
        # NVDA should be marked for liquidation (target = 0)
        assert allocation.get('NVDA', 0) == 0
        
        # Core ETFs should maintain target allocation
        assert allocation['VTI'] == 42000
        assert allocation['VXUS'] == 28000
        
        # Satellite stocks should split 30% equally
        expected_per_stock = 30000 / 2  # 15k each
        assert allocation['AAPL'] == expected_per_stock
        assert allocation['MSFT'] == expected_per_stock

    def test_generate_rebalancing_orders(self):
        """Test rebalancing order generation."""
        target_allocation = {
            'VTI': 42000,
            'AAPL': 15000,
            'MSFT': 15000,
            'GOOGL': 0  # Liquidate
        }
        
        current_holdings = {
            'VTI': {'shares': 100, 'market_value': 25000},
            'AAPL': {'shares': 50, 'market_value': 8000},
            'GOOGL': {'shares': 30, 'market_value': 5000}
        }
        
        current_prices = {
            'VTI': 250,
            'AAPL': 180,
            'MSFT': 300,
            'GOOGL': 150
        }
        
        orders = self.portfolio_manager.generate_rebalancing_orders(
            target_allocation, current_holdings, current_prices
        )
        
        # Check VTI buy order (need 42000 - 25000 = 17000 more)
        vti_order = next((o for o in orders if o['symbol'] == 'VTI'), None)
        assert vti_order is not None
        assert vti_order['side'] == 'buy'
        assert abs(vti_order['notional'] - 17000) < 1
        
        # Check GOOGL sell order (liquidate all)
        googl_order = next((o for o in orders if o['symbol'] == 'GOOGL'), None)
        assert googl_order is not None
        assert googl_order['side'] == 'sell'
        assert googl_order['qty'] == 30
        
        # Check AAPL buy order (need 15000 - 8000 = 7000 more)
        aapl_order = next((o for o in orders if o['symbol'] == 'AAPL'), None)
        assert aapl_order is not None
        assert aapl_order['side'] == 'buy'
        assert abs(aapl_order['notional'] - 7000) < 1
        
        # Check MSFT buy order (new position, need 15000)
        msft_order = next((o for o in orders if o['symbol'] == 'MSFT'), None)
        assert msft_order is not None
        assert msft_order['side'] == 'buy'
        assert abs(msft_order['notional'] - 15000) < 1

    def test_calculate_sell_orders_overvaluation(self):
        """Test sell order calculation for overvaluation."""
        current_holdings = {
            'AAPL': {'shares': 100, 'market_value': 20000}
        }
        
        current_prices = {'AAPL': 200}
        
        orders = self.portfolio_manager.calculate_sell_orders(
            ['AAPL'], 'overvaluation', current_holdings, current_prices
        )
        
        assert len(orders) == 1
        order = orders[0]
        assert order['symbol'] == 'AAPL'
        assert order['side'] == 'sell'
        assert order['qty'] == 50  # Trim to half
        assert order['reason'] == 'overvaluation'

    def test_calculate_sell_orders_broken_thesis(self):
        """Test sell order calculation for broken thesis."""
        current_holdings = {
            'AAPL': {'shares': 100, 'market_value': 20000}
        }
        
        current_prices = {'AAPL': 200}
        
        orders = self.portfolio_manager.calculate_sell_orders(
            ['AAPL'], 'broken_thesis', current_holdings, current_prices
        )
        
        assert len(orders) == 1
        order = orders[0]
        assert order['symbol'] == 'AAPL'
        assert order['side'] == 'sell'
        assert order['qty'] == 100  # Liquidate all
        assert order['reason'] == 'broken_thesis'

    def test_calculate_sell_orders_concentration(self):
        """Test sell order calculation for concentration."""
        current_holdings = {
            'AAPL': {'shares': 100, 'market_value': 25000}  # 25% of 100k portfolio
        }
        
        current_prices = {'AAPL': 250}
        portfolio_value = 100000
        
        orders = self.portfolio_manager.calculate_sell_orders(
            ['AAPL'], 'concentration', current_holdings, current_prices, portfolio_value
        )
        
        assert len(orders) == 1
        order = orders[0]
        assert order['symbol'] == 'AAPL'
        assert order['side'] == 'sell'
        
        # Should trim from 25% to 15% (target concentration)
        # Need to sell 10% of portfolio = 10k worth = 40 shares
        assert order['qty'] == 40
        assert order['reason'] == 'concentration'

    def test_update_holdings_after_orders(self):
        """Test holdings update after order execution."""
        current_holdings = {
            'AAPL': {'shares': 100, 'market_value': 20000}
        }
        
        executed_orders = [
            {
                'symbol': 'AAPL',
                'side': 'sell',
                'qty': 50,
                'filled_avg_price': 200
            },
            {
                'symbol': 'MSFT',
                'side': 'buy',
                'qty': 30,
                'filled_avg_price': 300
            }
        ]
        
        current_prices = {'AAPL': 200, 'MSFT': 300}
        
        updated_holdings = self.portfolio_manager.update_holdings_after_orders(
            current_holdings, executed_orders, current_prices
        )
        
        # AAPL should have 50 shares remaining
        assert updated_holdings['AAPL']['shares'] == 50
        assert updated_holdings['AAPL']['market_value'] == 10000
        
        # MSFT should be new position
        assert updated_holdings['MSFT']['shares'] == 30
        assert updated_holdings['MSFT']['market_value'] == 9000

    def test_calculate_portfolio_weights(self):
        """Test portfolio weight calculation."""
        holdings = {
            'VTI': {'shares': 100, 'market_value': 40000},
            'AAPL': {'shares': 50, 'market_value': 15000},
            'MSFT': {'shares': 30, 'market_value': 10000}
        }
        
        portfolio_value = 65000
        
        weights = self.portfolio_manager.calculate_portfolio_weights(
            holdings, portfolio_value
        )
        
        assert abs(weights['VTI'] - 0.615) < 0.01  # 40k/65k
        assert abs(weights['AAPL'] - 0.231) < 0.01  # 15k/65k
        assert abs(weights['MSFT'] - 0.154) < 0.01  # 10k/65k

    def test_get_portfolio_summary(self):
        """Test portfolio summary generation."""
        holdings = {
            'VTI': {'shares': 100, 'market_value': 40000},
            'AAPL': {'shares': 50, 'market_value': 15000},
            'MSFT': {'shares': 30, 'market_value': 10000}
        }
        
        cash_balance = 5000
        
        summary = self.portfolio_manager.get_portfolio_summary(holdings, cash_balance)
        
        assert summary['total_value'] == 70000
        assert summary['cash_balance'] == 5000
        assert summary['invested_value'] == 65000
        assert len(summary['holdings']) == 3
        assert summary['core_allocation'] == 40000 / 65000  # Only VTI is core
        assert summary['satellite_allocation'] == 25000 / 65000  # AAPL + MSFT

    def test_needs_rebalancing(self):
        """Test rebalancing need detection."""
        # Case 1: Needs rebalancing (core allocation off by more than threshold)
        holdings_off = {
            'VTI': {'shares': 100, 'market_value': 50000},  # 50% instead of 42%
            'AAPL': {'shares': 50, 'market_value': 15000}
        }
        
        assert self.portfolio_manager.needs_rebalancing(holdings_off, 65000) is True
        
        # Case 2: No rebalancing needed
        holdings_ok = {
            'VTI': {'shares': 100, 'market_value': 42000},  # Close to 42% target
            'VXUS': {'shares': 50, 'market_value': 28000},  # Close to 28% target
            'AAPL': {'shares': 50, 'market_value': 15000}
        }
        
        assert self.portfolio_manager.needs_rebalancing(holdings_ok, 85000) is False

    def test_calculate_dividend_reinvestment_orders(self):
        """Test dividend reinvestment order calculation."""
        dividend_data = [
            {'symbol': 'VTI', 'amount': 500},
            {'symbol': 'AAPL', 'amount': 200}
        ]
        
        current_prices = {'VTI': 250, 'AAPL': 180}
        
        orders = self.portfolio_manager.calculate_dividend_reinvestment_orders(
            dividend_data, current_prices
        )
        
        assert len(orders) == 2
        
        vti_order = next((o for o in orders if o['symbol'] == 'VTI'), None)
        assert vti_order['side'] == 'buy'
        assert vti_order['notional'] == 500
        
        aapl_order = next((o for o in orders if o['symbol'] == 'AAPL'), None)
        assert aapl_order['side'] == 'buy'
        assert aapl_order['notional'] == 200

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        # Mock historical returns
        returns_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=252, freq='D'),
            'portfolio_return': np.random.normal(0.0008, 0.02, 252)  # ~20% annual vol
        })
        
        benchmark_returns = np.random.normal(0.0007, 0.015, 252)  # ~15% annual vol
        
        metrics = self.portfolio_manager.calculate_risk_metrics(
            returns_data, benchmark_returns
        )
        
        assert 'annual_return' in metrics
        assert 'annual_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'alpha' in metrics
        assert 'beta' in metrics
        
        # Basic sanity checks
        assert -1 <= metrics['annual_return'] <= 2  # Reasonable return range
        assert 0 <= metrics['annual_volatility'] <= 1  # Reasonable volatility
        assert -5 <= metrics['sharpe_ratio'] <= 5  # Reasonable Sharpe ratio
        assert -1 <= metrics['max_drawdown'] <= 0  # Drawdown should be negative

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty selected stocks
        empty_stocks = pd.DataFrame()
        allocation = self.portfolio_manager.calculate_target_allocation(
            empty_stocks, 100000
        )
        
        # Should only have core ETF allocations
        assert 'VTI' in allocation
        assert 'VXUS' in allocation
        assert len(allocation) == 2
        
        # Zero portfolio value
        allocation_zero = self.portfolio_manager.calculate_target_allocation(
            empty_stocks, 0
        )
        assert all(v == 0 for v in allocation_zero.values())
        
        # Missing price data
        target_allocation = {'AAPL': 10000}
        current_holdings = {}
        current_prices = {}  # Missing AAPL price
        
        orders = self.portfolio_manager.generate_rebalancing_orders(
            target_allocation, current_holdings, current_prices
        )
        
        # Should skip orders for missing prices
        assert len(orders) == 0

    def test_minimum_trade_amount(self):
        """Test minimum trade amount filtering."""
        target_allocation = {'AAPL': 10050}  # Small difference
        current_holdings = {'AAPL': {'shares': 50, 'market_value': 10000}}
        current_prices = {'AAPL': 200}
        
        orders = self.portfolio_manager.generate_rebalancing_orders(
            target_allocation, current_holdings, current_prices
        )
        
        # Should not generate order for small amount (50 < 100 min trade)
        assert len(orders) == 0
        
        # Test with larger difference
        target_allocation = {'AAPL': 10200}  # Larger difference
        
        orders = self.portfolio_manager.generate_rebalancing_orders(
            target_allocation, current_holdings, current_prices
        )
        
        # Should generate order for larger amount (200 >= 100 min trade)
        assert len(orders) == 1

    def test_fractional_shares(self):
        """Test fractional share handling."""
        target_allocation = {'AAPL': 10000}
        current_holdings = {}
        current_prices = {'AAPL': 333.33}  # Price that would result in fractional shares
        
        orders = self.portfolio_manager.generate_rebalancing_orders(
            target_allocation, current_holdings, current_prices
        )
        
        assert len(orders) == 1
        order = orders[0]
        
        # Should use notional amount for fractional shares
        assert 'notional' in order
        assert order['notional'] == 10000
        assert order['type'] == 'market'
        assert order['time_in_force'] == 'day'