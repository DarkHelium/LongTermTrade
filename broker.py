"""Broker module for Alpaca API integration.

This module provides a wrapper for the Alpaca API to handle order management,
position tracking, and account information.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from alpaca_trade_api import REST, TimeFrame
from alpaca_trade_api.common import URL
from alpaca_trade_api.entity import Order, Position

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """Alpaca broker interface for trading operations."""
    
    def __init__(self, config: Dict):
        """Initialize Alpaca broker with configuration.
        
        Args:
            config: Configuration dictionary with Alpaca credentials
        """
        self.config = config
        alpaca_config = config.get('alpaca', {})
        
        self.api_key = alpaca_config.get('api_key', '')
        self.secret_key = alpaca_config.get('secret_key', '')
        self.base_url = alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')
        self.paper_trading = alpaca_config.get('paper_trading', True)
        
        # Initialize Alpaca REST client
        try:
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=URL(self.base_url),
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca - Account: {account.id}, Paper: {self.paper_trading}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca connection: {e}")
            raise
            
        # Trading parameters
        self.fractional_shares = config.get('trading', {}).get('fractional_shares', True)
        self.order_type = config.get('trading', {}).get('order_type', 'limit_at_open')
        self.slippage_tolerance = config.get('trading', {}).get('slippage_tolerance', 0.005)
        
    def get_account_info(self) -> Dict:
        """Get account information.
        
        Returns:
            Dictionary with account details
        """
        try:
            account = self.api.get_account()
            
            account_info = {
                'account_id': account.id,
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'multiplier': float(account.multiplier),
                'day_trade_count': int(account.day_trade_count),
                'daytrade_buying_power': float(account.daytrade_buying_power),
                'regt_buying_power': float(account.regt_buying_power),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'trade_suspended_by_user': account.trade_suspended_by_user,
                'paper_trading': self.paper_trading
            }
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
            
    def get_positions(self) -> List[Dict]:
        """Get current positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            positions = self.api.list_positions()
            
            position_list = []
            for position in positions:
                pos_dict = {
                    'symbol': position.symbol,
                    'shares': float(position.qty),
                    'side': position.side,
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price) if position.current_price else 0,
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'unrealized_intraday_pl': float(position.unrealized_intraday_pl),
                    'unrealized_intraday_plpc': float(position.unrealized_intraday_plpc),
                    'asset_id': position.asset_id,
                    'asset_class': position.asset_class,
                    'exchange': position.exchange
                }
                position_list.append(pos_dict)
                
            logger.info(f"Retrieved {len(position_list)} positions")
            return position_list
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
            
    def get_orders(self, status: str = 'all', limit: int = 100) -> List[Dict]:
        """Get order history.
        
        Args:
            status: Order status filter ('all', 'open', 'closed')
            limit: Maximum number of orders to retrieve
            
        Returns:
            List of order dictionaries
        """
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            
            order_list = []
            for order in orders:
                order_dict = {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'order_type': order.order_type,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty),
                    'status': order.status,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'avg_fill_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'time_in_force': order.time_in_force,
                    'created_at': order.created_at,
                    'updated_at': order.updated_at,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'expired_at': order.expired_at,
                    'canceled_at': order.canceled_at,
                    'asset_id': order.asset_id,
                    'asset_class': order.asset_class
                }
                order_list.append(order_dict)
                
            return order_list
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
            
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = 'market', limit_price: Optional[float] = None,
                   time_in_force: str = 'day', extended_hours: bool = False) -> Optional[Dict]:
        """Place a trading order.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares (can be fractional)
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            limit_price: Limit price for limit orders
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            extended_hours: Whether to allow extended hours trading
            
        Returns:
            Order dictionary if successful, None otherwise
        """
        try:
            # Validate inputs
            if not symbol or not side or quantity <= 0:
                logger.error(f"Invalid order parameters: {symbol}, {side}, {quantity}")
                return None
                
            # Check if fractional shares are supported
            if not self.fractional_shares and quantity != int(quantity):
                logger.warning(f"Fractional shares not enabled, rounding {quantity} to {int(quantity)}")
                quantity = int(quantity)
                if quantity == 0:
                    logger.warning("Rounded quantity is 0, skipping order")
                    return None
                    
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force,
                'extended_hours': extended_hours
            }
            
            # Add quantity (fractional or notional)
            if self.fractional_shares and quantity < 1:
                # Use notional for small fractional orders
                current_price = self.get_current_price(symbol)
                if current_price:
                    notional_value = quantity * current_price
                    order_params['notional'] = notional_value
                else:
                    order_params['qty'] = quantity
            else:
                order_params['qty'] = quantity
                
            # Add limit price if specified
            if order_type in ['limit', 'stop_limit'] and limit_price:
                order_params['limit_price'] = limit_price
                
            # Submit order
            order = self.api.submit_order(**order_params)
            
            order_dict = {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'order_type': order.order_type,
                'qty': float(order.qty) if order.qty else 0,
                'status': order.status,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'time_in_force': order.time_in_force,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at
            }
            
            logger.info(f"Order placed: {side} {quantity} {symbol} at {order_type} - ID: {order.id}")
            return order_dict
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
            
    def place_orders_batch(self, orders: List[Dict]) -> List[Dict]:
        """Place multiple orders with rate limiting.
        
        Args:
            orders: List of order dictionaries
            
        Returns:
            List of placed order results
        """
        results = []
        
        for i, order in enumerate(orders):
            try:
                # Rate limiting - wait between orders
                if i > 0:
                    time.sleep(0.5)  # 500ms between orders
                    
                result = self.place_order(
                    symbol=order.get('symbol', ''),
                    side=order.get('side', ''),
                    quantity=order.get('quantity', 0),
                    order_type=order.get('order_type', 'market'),
                    limit_price=order.get('limit_price'),
                    time_in_force=order.get('time_in_force', 'day')
                )
                
                if result:
                    result['original_order'] = order
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error in batch order {i}: {e}")
                continue
                
        logger.info(f"Placed {len(results)}/{len(orders)} orders successfully")
        return results
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Canceled order: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False
            
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders.
        
        Returns:
            True if successful
        """
        try:
            self.api.cancel_all_orders()
            logger.info("Canceled all open orders")
            return True
            
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            return False
            
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if failed
        """
        try:
            # Try to get latest trade first
            latest_trade = self.api.get_latest_trade(symbol)
            if latest_trade and latest_trade.price:
                return float(latest_trade.price)
                
            # Fallback to latest quote
            latest_quote = self.api.get_latest_quote(symbol)
            if latest_quote:
                bid = float(latest_quote.bid_price) if latest_quote.bid_price else 0
                ask = float(latest_quote.ask_price) if latest_quote.ask_price else 0
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
                elif bid > 0:
                    return bid
                elif ask > 0:
                    return ask
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
            
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to prices
        """
        prices = {}
        
        # Batch request for efficiency
        try:
            # Get latest trades for all symbols
            latest_trades = self.api.get_latest_trades(symbols)
            
            for symbol in symbols:
                if symbol in latest_trades and latest_trades[symbol].price:
                    prices[symbol] = float(latest_trades[symbol].price)
                else:
                    # Fallback to individual request
                    price = self.get_current_price(symbol)
                    if price:
                        prices[symbol] = price
                        
        except Exception as e:
            logger.warning(f"Batch price request failed, using individual requests: {e}")
            
            # Fallback to individual requests
            for symbol in symbols:
                price = self.get_current_price(symbol)
                if price:
                    prices[symbol] = price
                    
        logger.info(f"Retrieved prices for {len(prices)}/{len(symbols)} symbols")
        return prices
        
    def get_historical_data(self, symbol: str, timeframe: str = '1Day', 
                          start: Optional[datetime] = None, 
                          end: Optional[datetime] = None,
                          limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical price data.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe ('1Min', '5Min', '15Min', '30Min', '1Hour', '1Day')
            start: Start date
            end: End date
            limit: Maximum number of bars
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframe string to Alpaca TimeFrame
            timeframe_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame(5, TimeFrame.Unit.Minute),
                '15Min': TimeFrame(15, TimeFrame.Unit.Minute),
                '30Min': TimeFrame(30, TimeFrame.Unit.Minute),
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }
            
            tf = timeframe_map.get(timeframe, TimeFrame.Day)
            
            # Set default date range if not provided
            if not end:
                end = datetime.now()
            if not start:
                start = end - timedelta(days=365)  # 1 year default
                
            # Get bars
            bars = self.api.get_bars(
                symbol,
                tf,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=limit
            ).df
            
            if bars.empty:
                return None
                
            # Rename columns to standard format
            bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'TradeCount', 'VWAP']
            
            return bars[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
            
    def get_portfolio_history(self, period: str = '1M', timeframe: str = '1D') -> Optional[pd.DataFrame]:
        """Get portfolio performance history.
        
        Args:
            period: Time period ('1D', '1W', '1M', '3M', '1A', 'all')
            timeframe: Data resolution ('1Min', '5Min', '15Min', '1H', '1D')
            
        Returns:
            DataFrame with portfolio history
        """
        try:
            portfolio_history = self.api.get_portfolio_history(
                period=period,
                timeframe=timeframe
            )
            
            if not portfolio_history.equity:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame({
                'equity': portfolio_history.equity,
                'profit_loss': portfolio_history.profit_loss,
                'profit_loss_pct': portfolio_history.profit_loss_pct,
                'base_value': portfolio_history.base_value
            }, index=pd.to_datetime(portfolio_history.timestamp, unit='s'))
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return None
            
    def get_market_calendar(self, start: Optional[datetime] = None, 
                          end: Optional[datetime] = None) -> List[Dict]:
        """Get market calendar information.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List of market calendar entries
        """
        try:
            if not start:
                start = datetime.now()
            if not end:
                end = start + timedelta(days=30)
                
            calendar = self.api.get_calendar(
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d')
            )
            
            calendar_list = []
            for day in calendar:
                calendar_list.append({
                    'date': day.date,
                    'open': day.open,
                    'close': day.close
                })
                
            return calendar_list
            
        except Exception as e:
            logger.error(f"Error getting market calendar: {e}")
            return []
            
    def is_market_open(self) -> bool:
        """Check if market is currently open.
        
        Returns:
            True if market is open
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
            
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
            
    def wait_for_market_open(self, max_wait_minutes: int = 60):
        """Wait for market to open.
        
        Args:
            max_wait_minutes: Maximum time to wait in minutes
        """
        start_time = datetime.now()
        max_wait_time = start_time + timedelta(minutes=max_wait_minutes)
        
        while datetime.now() < max_wait_time:
            if self.is_market_open():
                logger.info("Market is now open")
                return
                
            logger.info("Waiting for market to open...")
            time.sleep(60)  # Check every minute
            
        logger.warning(f"Market did not open within {max_wait_minutes} minutes")
        
    def get_trading_summary(self) -> Dict:
        """Get trading summary for the session.
        
        Returns:
            Trading summary dictionary
        """
        try:
            account_info = self.get_account_info()
            positions = self.get_positions()
            recent_orders = self.get_orders(status='all', limit=50)
            
            # Calculate summary metrics
            total_positions = len(positions)
            total_market_value = sum(pos.get('market_value', 0) for pos in positions)
            total_unrealized_pl = sum(pos.get('unrealized_pl', 0) for pos in positions)
            
            # Count orders by status
            order_counts = {}
            for order in recent_orders:
                status = order.get('status', 'unknown')
                order_counts[status] = order_counts.get(status, 0) + 1
                
            summary = {
                'timestamp': datetime.now().isoformat(),
                'account_value': account_info.get('portfolio_value', 0),
                'cash_balance': account_info.get('cash', 0),
                'buying_power': account_info.get('buying_power', 0),
                'total_positions': total_positions,
                'total_market_value': total_market_value,
                'total_unrealized_pl': total_unrealized_pl,
                'unrealized_pl_pct': (total_unrealized_pl / total_market_value * 100) if total_market_value > 0 else 0,
                'order_counts': order_counts,
                'paper_trading': self.paper_trading,
                'market_open': self.is_market_open()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating trading summary: {e}")
            return {}