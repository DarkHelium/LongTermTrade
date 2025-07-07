"""Portfolio management module for allocation and rebalancing.

This module handles portfolio construction, rebalancing between core ETFs
and satellite stocks, and position sizing calculations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Portfolio management for long-term investment strategy."""
    
    def __init__(self, config: Dict):
        """Initialize portfolio manager with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.allocation = config.get('allocation', {})
        self.core_etfs = config.get('core_etfs', {})
        self.stock_selection = config.get('stock_selection', {})
        
        # Portfolio state
        self.current_holdings = []
        self.cash_balance = 0.0
        self.total_portfolio_value = 0.0
        
    def calculate_target_allocation(self, selected_stocks: List[Dict], 
                                  current_prices: Dict[str, float],
                                  total_portfolio_value: float) -> Dict[str, Dict]:
        """Calculate target allocation for portfolio.
        
        Args:
            selected_stocks: List of selected satellite stocks
            current_prices: Dictionary of current prices
            total_portfolio_value: Total portfolio value
            
        Returns:
            Dictionary with target allocations for each symbol
        """
        target_allocation = {}
        
        # Core ETF allocation (70% by default)
        core_allocation = self.allocation.get('core_etfs', 0.70)
        core_value = total_portfolio_value * core_allocation
        
        # Allocate among core ETFs
        for etf_symbol, etf_weight in self.core_etfs.items():
            etf_value = core_value * etf_weight
            current_price = current_prices.get(etf_symbol, 0)
            
            if current_price > 0:
                target_shares = etf_value / current_price
                target_allocation[etf_symbol] = {
                    'target_shares': target_shares,
                    'target_value': etf_value,
                    'target_weight': core_allocation * etf_weight,
                    'asset_type': 'core_etf'
                }
                
        # Satellite stock allocation (30% by default)
        satellite_allocation = self.allocation.get('satellite_stocks', 0.30)
        satellite_value = total_portfolio_value * satellite_allocation
        
        # Equal weight among selected stocks
        if selected_stocks:
            stock_weight = satellite_allocation / len(selected_stocks)
            stock_value = satellite_value / len(selected_stocks)
            
            for stock in selected_stocks:
                symbol = stock.get('symbol', '')
                current_price = current_prices.get(symbol, 0)
                
                if current_price > 0:
                    target_shares = stock_value / current_price
                    target_allocation[symbol] = {
                        'target_shares': target_shares,
                        'target_value': stock_value,
                        'target_weight': stock_weight,
                        'asset_type': 'satellite_stock',
                        'composite_score': stock.get('composite_score', 0)
                    }
                    
        logger.info(f"Calculated target allocation for {len(target_allocation)} positions")
        return target_allocation
        
    def calculate_rebalancing_orders(self, current_holdings: List[Dict],
                                   target_allocation: Dict[str, Dict],
                                   current_prices: Dict[str, float]) -> List[Dict]:
        """Calculate orders needed for rebalancing.
        
        Args:
            current_holdings: Current portfolio holdings
            target_allocation: Target allocation dictionary
            current_prices: Current prices
            
        Returns:
            List of order dictionaries
        """
        orders = []
        
        # Create lookup for current holdings
        current_lookup = {h.get('symbol', ''): h for h in current_holdings}
        
        # Calculate orders for each target position
        for symbol, target in target_allocation.items():
            current_holding = current_lookup.get(symbol, {})
            current_shares = current_holding.get('shares', 0)
            target_shares = target.get('target_shares', 0)
            
            shares_diff = target_shares - current_shares
            
            # Only create order if difference is significant
            min_order_value = 100  # Minimum $100 order
            current_price = current_prices.get(symbol, 0)
            
            if abs(shares_diff * current_price) >= min_order_value:
                order = {
                    'symbol': symbol,
                    'side': 'buy' if shares_diff > 0 else 'sell',
                    'quantity': abs(shares_diff),
                    'order_type': 'limit',
                    'limit_price': self._calculate_limit_price(symbol, current_price, shares_diff > 0),
                    'asset_type': target.get('asset_type', 'unknown'),
                    'reason': 'rebalance'
                }
                orders.append(order)
                
        # Check for positions to liquidate (not in target allocation)
        for symbol, holding in current_lookup.items():
            if symbol not in target_allocation and holding.get('shares', 0) > 0:
                order = {
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': holding.get('shares', 0),
                    'order_type': 'market',
                    'asset_type': holding.get('asset_type', 'unknown'),
                    'reason': 'liquidate'
                }
                orders.append(order)
                
        logger.info(f"Generated {len(orders)} rebalancing orders")
        return orders
        
    def calculate_sell_orders(self, sell_decisions: Dict[str, str],
                            current_holdings: List[Dict],
                            current_prices: Dict[str, float]) -> List[Dict]:
        """Calculate sell orders based on sell decisions.
        
        Args:
            sell_decisions: Dictionary mapping symbols to sell reasons
            current_holdings: Current portfolio holdings
            current_prices: Current prices
            
        Returns:
            List of sell order dictionaries
        """
        orders = []
        current_lookup = {h.get('symbol', ''): h for h in current_holdings}
        
        for symbol, reason in sell_decisions.items():
            holding = current_lookup.get(symbol)
            if not holding or holding.get('shares', 0) <= 0:
                continue
                
            current_shares = holding.get('shares', 0)
            current_price = current_prices.get(symbol, 0)
            
            if reason == 'overvaluation':
                # Trim to half position
                trim_percentage = self.config.get('sell_rules', {}).get('overvaluation', {}).get('trim_percentage', 0.50)
                sell_quantity = current_shares * (1 - trim_percentage)
            elif reason == 'concentration':
                # Trim to target weight
                target_weight = self.config.get('sell_rules', {}).get('concentration', {}).get('trim_target_weight', 0.15)
                current_value = current_shares * current_price
                target_value = self.total_portfolio_value * target_weight
                if current_value > target_value:
                    target_shares = target_value / current_price if current_price > 0 else 0
                    sell_quantity = current_shares - target_shares
                else:
                    continue
            else:
                # Full liquidation for broken thesis or replacement
                sell_quantity = current_shares
                
            if sell_quantity > 0:
                order = {
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': sell_quantity,
                    'order_type': 'limit',
                    'limit_price': self._calculate_limit_price(symbol, current_price, False),
                    'asset_type': holding.get('asset_type', 'unknown'),
                    'reason': reason
                }
                orders.append(order)
                
        logger.info(f"Generated {len(orders)} sell orders")
        return orders
        
    def _calculate_limit_price(self, symbol: str, current_price: float, is_buy: bool) -> float:
        """Calculate limit price with slippage tolerance.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Limit price
        """
        slippage_tolerance = self.config.get('trading', {}).get('slippage_tolerance', 0.005)
        
        if is_buy:
            # Buy slightly above market
            return current_price * (1 + slippage_tolerance)
        else:
            # Sell slightly below market
            return current_price * (1 - slippage_tolerance)
            
    def update_holdings(self, executed_orders: List[Dict], current_prices: Dict[str, float]):
        """Update portfolio holdings after order execution.
        
        Args:
            executed_orders: List of executed orders
            current_prices: Current market prices
        """
        holdings_lookup = {h.get('symbol', ''): h for h in self.current_holdings}
        
        for order in executed_orders:
            symbol = order.get('symbol', '')
            side = order.get('side', '')
            quantity = order.get('filled_quantity', 0)
            price = order.get('fill_price', 0)
            
            if quantity <= 0:
                continue
                
            if symbol not in holdings_lookup:
                holdings_lookup[symbol] = {
                    'symbol': symbol,
                    'shares': 0,
                    'avg_cost': 0,
                    'total_cost': 0,
                    'asset_type': order.get('asset_type', 'unknown')
                }
                
            holding = holdings_lookup[symbol]
            
            if side == 'buy':
                # Update position for buy
                old_shares = holding['shares']
                old_cost = holding['total_cost']
                new_cost = quantity * price
                
                holding['shares'] = old_shares + quantity
                holding['total_cost'] = old_cost + new_cost
                holding['avg_cost'] = holding['total_cost'] / holding['shares'] if holding['shares'] > 0 else 0
                
            elif side == 'sell':
                # Update position for sell
                holding['shares'] = max(0, holding['shares'] - quantity)
                if holding['shares'] == 0:
                    holding['total_cost'] = 0
                    holding['avg_cost'] = 0
                else:
                    # Proportionally reduce total cost
                    cost_reduction = (quantity / (holding['shares'] + quantity)) * holding['total_cost']
                    holding['total_cost'] = max(0, holding['total_cost'] - cost_reduction)
                    holding['avg_cost'] = holding['total_cost'] / holding['shares']
                    
        # Remove positions with zero shares
        self.current_holdings = [h for h in holdings_lookup.values() if h.get('shares', 0) > 0]
        
        # Update portfolio value
        self._update_portfolio_value(current_prices)
        
    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update total portfolio value."""
        total_value = self.cash_balance
        
        for holding in self.current_holdings:
            symbol = holding.get('symbol', '')
            shares = holding.get('shares', 0)
            current_price = current_prices.get(symbol, 0)
            total_value += shares * current_price
            
        self.total_portfolio_value = total_value
        
    def calculate_portfolio_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate current portfolio weights.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Dictionary mapping symbols to portfolio weights
        """
        if self.total_portfolio_value <= 0:
            return {}
            
        weights = {}
        for holding in self.current_holdings:
            symbol = holding.get('symbol', '')
            shares = holding.get('shares', 0)
            current_price = current_prices.get(symbol, 0)
            position_value = shares * current_price
            weights[symbol] = position_value / self.total_portfolio_value
            
        return weights
        
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Generate portfolio summary.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Portfolio summary dictionary
        """
        self._update_portfolio_value(current_prices)
        weights = self.calculate_portfolio_weights(current_prices)
        
        # Categorize holdings
        core_etf_value = 0
        satellite_stock_value = 0
        
        holdings_detail = []
        for holding in self.current_holdings:
            symbol = holding.get('symbol', '')
            shares = holding.get('shares', 0)
            avg_cost = holding.get('avg_cost', 0)
            current_price = current_prices.get(symbol, 0)
            position_value = shares * current_price
            unrealized_pnl = position_value - (shares * avg_cost)
            
            holding_detail = {
                'symbol': symbol,
                'shares': shares,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'position_value': position_value,
                'weight': weights.get(symbol, 0),
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / (shares * avg_cost)) if shares * avg_cost > 0 else 0,
                'asset_type': holding.get('asset_type', 'unknown')
            }
            holdings_detail.append(holding_detail)
            
            # Categorize by asset type
            if holding.get('asset_type') == 'core_etf':
                core_etf_value += position_value
            elif holding.get('asset_type') == 'satellite_stock':
                satellite_stock_value += position_value
                
        # Sort by position value
        holdings_detail.sort(key=lambda x: x['position_value'], reverse=True)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_portfolio_value': self.total_portfolio_value,
            'cash_balance': self.cash_balance,
            'invested_value': self.total_portfolio_value - self.cash_balance,
            'core_etf_value': core_etf_value,
            'satellite_stock_value': satellite_stock_value,
            'core_etf_allocation': core_etf_value / self.total_portfolio_value if self.total_portfolio_value > 0 else 0,
            'satellite_allocation': satellite_stock_value / self.total_portfolio_value if self.total_portfolio_value > 0 else 0,
            'cash_allocation': self.cash_balance / self.total_portfolio_value if self.total_portfolio_value > 0 else 0,
            'num_positions': len(self.current_holdings),
            'holdings': holdings_detail
        }
        
        return summary
        
    def check_rebalancing_needed(self, current_prices: Dict[str, float]) -> bool:
        """Check if portfolio rebalancing is needed.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            True if rebalancing is needed
        """
        weights = self.calculate_portfolio_weights(current_prices)
        target_core_allocation = self.allocation.get('core_etfs', 0.70)
        target_satellite_allocation = self.allocation.get('satellite_stocks', 0.30)
        
        # Calculate current allocations
        current_core_allocation = 0
        current_satellite_allocation = 0
        
        for holding in self.current_holdings:
            symbol = holding.get('symbol', '')
            weight = weights.get(symbol, 0)
            
            if holding.get('asset_type') == 'core_etf':
                current_core_allocation += weight
            elif holding.get('asset_type') == 'satellite_stock':
                current_satellite_allocation += weight
                
        # Check if allocations are significantly off target
        allocation_tolerance = 0.05  # 5% tolerance
        
        core_drift = abs(current_core_allocation - target_core_allocation)
        satellite_drift = abs(current_satellite_allocation - target_satellite_allocation)
        
        needs_rebalancing = (core_drift > allocation_tolerance or 
                           satellite_drift > allocation_tolerance)
        
        if needs_rebalancing:
            logger.info(f"Rebalancing needed - Core: {current_core_allocation:.1%} vs {target_core_allocation:.1%}, "
                       f"Satellite: {current_satellite_allocation:.1%} vs {target_satellite_allocation:.1%}")
                       
        return needs_rebalancing
        
    def calculate_dividend_reinvestment(self, dividend_payments: List[Dict],
                                      current_prices: Dict[str, float]) -> List[Dict]:
        """Calculate orders for dividend reinvestment.
        
        Args:
            dividend_payments: List of dividend payment records
            current_prices: Current market prices
            
        Returns:
            List of reinvestment orders
        """
        if not self.config.get('trading', {}).get('reinvest_dividends', True):
            return []
            
        orders = []
        total_dividends = sum(d.get('amount', 0) for d in dividend_payments)
        
        if total_dividends < 50:  # Minimum reinvestment amount
            return []
            
        # Reinvest proportionally to target allocation
        target_core_allocation = self.allocation.get('core_etfs', 0.70)
        core_reinvestment = total_dividends * target_core_allocation
        
        # Reinvest in primary core ETF
        primary_etf = list(self.core_etfs.keys())[0] if self.core_etfs else 'VTI'
        etf_price = current_prices.get(primary_etf, 0)
        
        if etf_price > 0 and core_reinvestment >= etf_price:
            shares_to_buy = core_reinvestment / etf_price
            order = {
                'symbol': primary_etf,
                'side': 'buy',
                'quantity': shares_to_buy,
                'order_type': 'market',
                'asset_type': 'core_etf',
                'reason': 'dividend_reinvestment'
            }
            orders.append(order)
            
        logger.info(f"Generated {len(orders)} dividend reinvestment orders for ${total_dividends:.2f}")
        return orders
        
    def get_risk_metrics(self, price_history: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate portfolio risk metrics.
        
        Args:
            price_history: Dictionary of price history DataFrames
            
        Returns:
            Risk metrics dictionary
        """
        if not self.current_holdings:
            return {}
            
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(price_history)
            
            if portfolio_returns.empty:
                return {}
                
            # Calculate risk metrics
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            risk_metrics = {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': np.percentile(portfolio_returns, 5),
                'var_99': np.percentile(portfolio_returns, 1)
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
            
    def _calculate_portfolio_returns(self, price_history: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculate portfolio returns from price history."""
        if not self.current_holdings:
            return pd.Series()
            
        # Get weights
        weights = {h.get('symbol', ''): h.get('shares', 0) for h in self.current_holdings}
        total_shares_value = sum(weights.values())
        
        if total_shares_value == 0:
            return pd.Series()
            
        # Normalize weights
        for symbol in weights:
            weights[symbol] = weights[symbol] / total_shares_value
            
        # Calculate weighted returns
        portfolio_returns = None
        
        for symbol, weight in weights.items():
            if symbol in price_history and weight > 0:
                stock_prices = price_history[symbol]['Close']
                stock_returns = stock_prices.pct_change().dropna()
                
                if portfolio_returns is None:
                    portfolio_returns = stock_returns * weight
                else:
                    # Align indices and add
                    aligned_returns = stock_returns.reindex(portfolio_returns.index, fill_value=0)
                    portfolio_returns += aligned_returns * weight
                    
        return portfolio_returns if portfolio_returns is not None else pd.Series()