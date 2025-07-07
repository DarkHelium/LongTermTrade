#!/usr/bin/env python3
"""Main module for the Long-Term Stock Investment Bot.

This module orchestrates the entire investment strategy, including:
- Monthly automated workflow
- Stock screening and scoring
- Portfolio rebalancing
- Sell/trim logic
- Optional sentiment overlay
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from broker import AlpacaBroker
from data import DataProvider
from portfolio import PortfolioManager
from scoring import FundamentalScorer
from sentiment import SentimentAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/longterm_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LongTermInvestmentBot:
    """Long-term stock investment bot with fundamental analysis."""
    
    def __init__(self, config_path: str = 'config.yml'):
        """Initialize the investment bot.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_provider = DataProvider(self.config)
        self.scorer = FundamentalScorer(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        self.broker = AlpacaBroker(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        
        # Bot state
        self.last_rebalance_date = None
        self.sentiment_history = []
        
        logger.info("Long-Term Investment Bot initialized")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def run_monthly_workflow(self, force_rebalance: bool = False) -> Dict:
        """Run the monthly investment workflow.
        
        Args:
            force_rebalance: Force rebalancing regardless of schedule
            
        Returns:
            Dictionary with workflow results
        """
        logger.info("Starting monthly workflow")
        
        try:
            # Check if rebalancing is needed
            if not force_rebalance and not self._should_rebalance():
                logger.info("Rebalancing not needed, skipping workflow")
                return {'status': 'skipped', 'reason': 'not_scheduled'}
                
            # Step 1: Get current portfolio state
            current_positions = self.broker.get_positions()
            account_info = self.broker.get_account_info()
            
            logger.info(f"Current portfolio value: ${account_info.get('portfolio_value', 0):,.2f}")
            
            # Step 2: Get stock universe and fundamentals
            logger.info("Fetching stock universe and fundamentals...")
            stock_universe = self.data_provider.get_stock_universe()
            
            if not stock_universe:
                logger.error("Failed to get stock universe")
                return {'status': 'error', 'reason': 'no_stock_data'}
                
            # Step 3: Score all stocks
            logger.info(f"Scoring {len(stock_universe)} stocks...")
            scored_stocks = self.scorer.score_stocks(stock_universe)
            
            if not scored_stocks:
                logger.error("No stocks passed screening")
                return {'status': 'error', 'reason': 'no_qualified_stocks'}
                
            # Step 4: Select top stocks
            num_stocks = self.config.get('portfolio', {}).get('num_satellite_stocks', 20)
            selected_stocks = self.scorer.select_top_stocks(scored_stocks, num_stocks)
            
            logger.info(f"Selected {len(selected_stocks)} top stocks")
            
            # Step 5: Check sell/trim conditions for current holdings
            sell_orders = self._check_sell_conditions(current_positions, scored_stocks)
            
            # Step 6: Update portfolio manager with current state
            self.portfolio_manager.update_holdings(current_positions)
            
            # Step 7: Calculate target allocation
            target_allocation = self.portfolio_manager.calculate_target_allocation(
                selected_stocks, account_info.get('portfolio_value', 0)
            )
            
            # Step 8: Generate rebalancing orders
            rebalancing_orders = self.portfolio_manager.generate_rebalancing_orders(
                target_allocation, current_positions
            )
            
            # Step 9: Apply sentiment overlay (if enabled)
            if self.sentiment_analyzer.is_enabled():
                rebalancing_orders = self._apply_sentiment_overlay(
                    rebalancing_orders, selected_stocks
                )
                
            # Step 10: Execute orders
            execution_results = self._execute_orders(sell_orders + rebalancing_orders)
            
            # Step 11: Update state
            self.last_rebalance_date = datetime.now().date()
            
            # Step 12: Generate summary
            workflow_summary = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'stocks_analyzed': len(stock_universe),
                'stocks_selected': len(selected_stocks),
                'sell_orders': len(sell_orders),
                'rebalancing_orders': len(rebalancing_orders),
                'orders_executed': len([r for r in execution_results if r is not None]),
                'portfolio_value': account_info.get('portfolio_value', 0),
                'selected_stocks': [s['symbol'] for s in selected_stocks],
                'execution_results': execution_results
            }
            
            logger.info(f"Monthly workflow completed: {workflow_summary['orders_executed']} orders executed")
            return workflow_summary
            
        except Exception as e:
            logger.error(f"Error in monthly workflow: {e}")
            return {'status': 'error', 'reason': str(e)}
            
    def _should_rebalance(self) -> bool:
        """Check if rebalancing is needed based on schedule.
        
        Returns:
            True if rebalancing is needed
        """
        if self.last_rebalance_date is None:
            return True
            
        # Check if it's the first trading day of the month
        today = datetime.now().date()
        
        # If we haven't rebalanced this month
        if (today.year, today.month) != (self.last_rebalance_date.year, self.last_rebalance_date.month):
            # Check if market is open and it's a trading day
            if self.broker.is_market_open():
                return True
                
        return False
        
    def _check_sell_conditions(self, current_positions: List[Dict], 
                             scored_stocks: List[Dict]) -> List[Dict]:
        """Check sell/trim conditions for current holdings.
        
        Args:
            current_positions: Current portfolio positions
            scored_stocks: All scored stocks
            
        Returns:
            List of sell orders
        """
        sell_orders = []
        
        # Create lookup for scored stocks
        scored_lookup = {stock['symbol']: stock for stock in scored_stocks}
        
        for position in current_positions:
            symbol = position['symbol']
            
            # Skip ETFs (core holdings)
            core_etfs = self.config.get('portfolio', {}).get('core_etfs', [])
            if symbol in [etf['symbol'] for etf in core_etfs]:
                continue
                
            try:
                # Check various sell conditions
                sell_reason = None
                sell_percentage = 1.0  # Full liquidation by default
                
                # 1. Broken Thesis
                if self.scorer.check_broken_thesis(symbol, scored_lookup.get(symbol)):
                    sell_reason = 'broken_thesis'
                    sell_percentage = 1.0
                    
                # 2. Extreme Overvaluation
                elif self.scorer.check_overvaluation(symbol, scored_lookup.get(symbol)):
                    sell_reason = 'overvaluation'
                    sell_percentage = 0.5  # Trim to half
                    
                # 3. Concentration Cap
                elif self.portfolio_manager.check_concentration_limit(symbol, position):
                    sell_reason = 'concentration'
                    # Calculate how much to trim to get to 15%
                    target_weight = 0.15
                    current_weight = position['market_value'] / self.portfolio_manager.total_portfolio_value
                    if current_weight > target_weight:
                        sell_percentage = 1 - (target_weight / current_weight)
                    else:
                        continue
                        
                # 4. Better Replacement
                elif self.scorer.check_better_replacement(symbol, scored_lookup.get(symbol), scored_stocks):
                    sell_reason = 'better_replacement'
                    sell_percentage = 1.0
                    
                # 5. Sentiment-accelerated selling (if enabled)
                elif (self.sentiment_analyzer.is_enabled() and 
                      self._check_sentiment_sell_condition(symbol)):
                    sell_reason = 'sentiment_accelerated'
                    sell_percentage = 0.5  # Trim to half
                    
                if sell_reason:
                    shares_to_sell = position['shares'] * sell_percentage
                    
                    sell_order = {
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': shares_to_sell,
                        'order_type': 'limit',
                        'reason': sell_reason,
                        'current_position': position
                    }
                    
                    # Set limit price with small discount
                    current_price = self.broker.get_current_price(symbol)
                    if current_price:
                        limit_price = current_price * 0.995  # 0.5% discount
                        sell_order['limit_price'] = limit_price
                        
                    sell_orders.append(sell_order)
                    
                    logger.info(f"Sell order generated: {sell_reason} - {symbol} ({shares_to_sell:.2f} shares)")
                    
            except Exception as e:
                logger.error(f"Error checking sell conditions for {symbol}: {e}")
                continue
                
        return sell_orders
        
    def _check_sentiment_sell_condition(self, symbol: str) -> bool:
        """Check if sentiment-based selling condition is met.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if sentiment sell condition is met
        """
        try:
            # Check if we have enough sentiment history
            if len(self.sentiment_history) < 10:
                return False
                
            # Check exuberance condition
            return self.sentiment_analyzer.check_exuberance_condition(
                symbol, self.sentiment_history
            )
            
        except Exception as e:
            logger.error(f"Error checking sentiment sell condition for {symbol}: {e}")
            return False
            
    def _apply_sentiment_overlay(self, orders: List[Dict], 
                               selected_stocks: List[Dict]) -> List[Dict]:
        """Apply sentiment overlay to modify orders.
        
        Args:
            orders: List of orders
            selected_stocks: Selected stocks
            
        Returns:
            Modified orders list
        """
        if not self.sentiment_analyzer.is_enabled():
            return orders
            
        try:
            modified_orders = []
            
            for order in orders:
                symbol = order.get('symbol', '')
                
                # Only apply to buy orders for satellite stocks
                if order.get('side') != 'buy':
                    modified_orders.append(order)
                    continue
                    
                # Get sentiment for this symbol
                sentiment_data = self.sentiment_analyzer.get_combined_sentiment(symbol)
                
                # Store sentiment data for history
                sentiment_entry = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'sentiment_score': sentiment_data['sentiment_score'],
                    'confidence': sentiment_data['confidence']
                }
                self.sentiment_history.append(sentiment_entry)
                
                # Keep only last 30 days of sentiment history
                cutoff_date = datetime.now() - timedelta(days=30)
                self.sentiment_history = [
                    entry for entry in self.sentiment_history 
                    if entry['timestamp'] > cutoff_date
                ]
                
                # Apply sentiment-based modifications
                if sentiment_data['confidence'] > 0.3:  # Only if we have confident data
                    sentiment_score = sentiment_data['sentiment_score']
                    
                    # Reduce buy orders for highly positive sentiment (contrarian)
                    if sentiment_score > 0.7:  # Very positive sentiment
                        order['quantity'] *= 0.5  # Reduce position size
                        order['sentiment_adjustment'] = 'reduced_bullish'
                        logger.info(f"Reduced buy order for {symbol} due to high positive sentiment")
                        
                    # Slightly increase buy orders for negative sentiment
                    elif sentiment_score < -0.3:  # Negative sentiment
                        order['quantity'] *= 1.2  # Increase position size slightly
                        order['sentiment_adjustment'] = 'increased_bearish'
                        logger.info(f"Increased buy order for {symbol} due to negative sentiment")
                        
                modified_orders.append(order)
                
            return modified_orders
            
        except Exception as e:
            logger.error(f"Error applying sentiment overlay: {e}")
            return orders
            
    def _execute_orders(self, orders: List[Dict]) -> List[Optional[Dict]]:
        """Execute a list of orders.
        
        Args:
            orders: List of order dictionaries
            
        Returns:
            List of execution results
        """
        if not orders:
            return []
            
        logger.info(f"Executing {len(orders)} orders")
        
        # Group orders by type for better execution
        sell_orders = [o for o in orders if o.get('side') == 'sell']
        buy_orders = [o for o in orders if o.get('side') == 'buy']
        
        results = []
        
        # Execute sell orders first
        if sell_orders:
            logger.info(f"Executing {len(sell_orders)} sell orders")
            sell_results = self.broker.place_orders_batch(sell_orders)
            results.extend(sell_results)
            
        # Wait a bit for sells to settle
        if sell_orders and buy_orders:
            import time
            time.sleep(2)
            
        # Execute buy orders
        if buy_orders:
            logger.info(f"Executing {len(buy_orders)} buy orders")
            buy_results = self.broker.place_orders_batch(buy_orders)
            results.extend(buy_results)
            
        return results
        
    def run_backtest(self, start_date: str, end_date: str, 
                    initial_capital: float = 100000) -> Dict:
        """Run a backtest of the strategy.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital amount
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        try:
            # Import backtest module
            from backtest import LongTermBacktester
            
            backtester = LongTermBacktester(self.config)
            
            results = backtester.run_backtest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                data_provider=self.data_provider,
                scorer=self.scorer,
                portfolio_manager=self.portfolio_manager
            )
            
            logger.info("Backtest completed")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'status': 'error', 'reason': str(e)}
            
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary.
        
        Returns:
            Portfolio summary dictionary
        """
        try:
            # Get current data
            account_info = self.broker.get_account_info()
            positions = self.broker.get_positions()
            
            # Update portfolio manager
            self.portfolio_manager.update_holdings(positions)
            
            # Generate summary
            summary = self.portfolio_manager.get_portfolio_summary()
            
            # Add account info
            summary.update({
                'account_value': account_info.get('portfolio_value', 0),
                'cash_balance': account_info.get('cash', 0),
                'buying_power': account_info.get('buying_power', 0),
                'paper_trading': account_info.get('paper_trading', True)
            })
            
            # Add risk metrics
            risk_metrics = self.portfolio_manager.calculate_risk_metrics()
            summary['risk_metrics'] = risk_metrics
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {'error': str(e)}
            
    def get_trading_summary(self) -> Dict:
        """Get trading summary from broker.
        
        Returns:
            Trading summary dictionary
        """
        return self.broker.get_trading_summary()
        
    def check_market_status(self) -> Dict:
        """Check market status and trading conditions.
        
        Returns:
            Market status dictionary
        """
        try:
            is_open = self.broker.is_market_open()
            calendar = self.broker.get_market_calendar()
            
            status = {
                'market_open': is_open,
                'timestamp': datetime.now().isoformat(),
                'next_trading_days': calendar[:5] if calendar else []
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return {'error': str(e)}


def main():
    """Main entry point for the bot."""
    parser = argparse.ArgumentParser(description='Long-Term Stock Investment Bot')
    parser.add_argument('--config', default='config.yml', help='Configuration file path')
    parser.add_argument('--backtest', action='store_true', help='Run backtest mode')
    parser.add_argument('--live', action='store_true', help='Run live trading mode')
    parser.add_argument('--paper', action='store_true', help='Run paper trading mode')
    parser.add_argument('--rebalance-now', action='store_true', help='Force immediate rebalancing')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital for backtest')
    parser.add_argument('--summary', action='store_true', help='Show portfolio summary')
    parser.add_argument('--status', action='store_true', help='Show market status')
    
    args = parser.parse_args()
    
    try:
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # Initialize bot
        bot = LongTermInvestmentBot(args.config)
        
        if args.backtest:
            # Run backtest
            if not args.start_date or not args.end_date:
                logger.error("Backtest requires --start-date and --end-date")
                sys.exit(1)
                
            results = bot.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital
            )
            
            print("\n=== BACKTEST RESULTS ===")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
                    
        elif args.summary:
            # Show portfolio summary
            summary = bot.get_portfolio_summary()
            
            print("\n=== PORTFOLIO SUMMARY ===")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    if 'value' in key.lower() or 'balance' in key.lower():
                        print(f"{key}: ${value:,.2f}")
                    elif 'pct' in key.lower() or 'percent' in key.lower():
                        print(f"{key}: {value:.2f}%")
                    else:
                        print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
                    
        elif args.status:
            # Show market status
            status = bot.check_market_status()
            
            print("\n=== MARKET STATUS ===")
            for key, value in status.items():
                print(f"{key}: {value}")
                
        elif args.live or args.paper or args.rebalance_now:
            # Run trading workflow
            if args.live:
                logger.warning("Live trading mode - real money at risk!")
                response = input("Are you sure you want to proceed with live trading? (yes/no): ")
                if response.lower() != 'yes':
                    logger.info("Live trading cancelled")
                    sys.exit(0)
                    
            # Run monthly workflow
            results = bot.run_monthly_workflow(force_rebalance=args.rebalance_now)
            
            print("\n=== WORKFLOW RESULTS ===")
            for key, value in results.items():
                if key != 'execution_results':  # Skip detailed execution results
                    print(f"{key}: {value}")
                    
            # Show trading summary
            trading_summary = bot.get_trading_summary()
            print("\n=== TRADING SUMMARY ===")
            for key, value in trading_summary.items():
                if isinstance(value, (int, float)):
                    if 'value' in key.lower() or 'balance' in key.lower():
                        print(f"{key}: ${value:,.2f}")
                    else:
                        print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
                    
        else:
            # Default: show help
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()