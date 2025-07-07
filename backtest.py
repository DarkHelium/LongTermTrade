"""Backtesting module for the Long-Term Investment Bot.

This module provides vectorized backtesting capabilities to evaluate
the performance of the long-term investment strategy over historical data.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LongTermBacktester:
    """Vectorized backtester for long-term investment strategy."""
    
    def __init__(self, config: Dict):
        """Initialize backtester with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Backtest parameters
        backtest_config = config.get('backtesting', {})
        self.rebalance_frequency = backtest_config.get('rebalance_frequency', 'monthly')
        self.transaction_cost = backtest_config.get('transaction_cost', 0.001)  # 0.1%
        self.slippage = backtest_config.get('slippage', 0.001)  # 0.1%
        
        # Portfolio parameters
        portfolio_config = config.get('portfolio', {})
        self.core_allocation = portfolio_config.get('core_allocation', 0.7)
        self.satellite_allocation = portfolio_config.get('satellite_allocation', 0.3)
        self.num_satellite_stocks = portfolio_config.get('num_satellite_stocks', 20)
        
        # Results storage
        self.results = {}
        self.trades = []
        self.portfolio_history = pd.DataFrame()
        
        logger.info("Backtester initialized")
        
    def run_backtest(self, start_date: str, end_date: str, initial_capital: float,
                    data_provider, scorer, portfolio_manager) -> Dict:
        """Run complete backtest of the strategy.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital amount
            data_provider: Data provider instance
            scorer: Fundamental scorer instance
            portfolio_manager: Portfolio manager instance
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        try:
            # Convert dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Generate rebalancing dates
            rebalance_dates = self._generate_rebalance_dates(start_dt, end_dt)
            
            logger.info(f"Generated {len(rebalance_dates)} rebalancing dates")
            
            # Initialize portfolio state
            portfolio_value = initial_capital
            cash = initial_capital
            holdings = {}  # symbol -> shares
            
            # Get benchmark data (SPY)
            benchmark_data = self._get_benchmark_data(start_date, end_date, data_provider)
            
            # Initialize results tracking
            portfolio_values = []
            benchmark_values = []
            dates = []
            
            # Get core ETF data
            core_etfs = self.config.get('portfolio', {}).get('core_etfs', [
                {'symbol': 'VTI', 'weight': 0.6},
                {'symbol': 'VXUS', 'weight': 0.4}
            ])
            
            core_etf_data = {}
            for etf in core_etfs:
                symbol = etf['symbol']
                data = data_provider.get_historical_prices(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    core_etf_data[symbol] = data
                    
            logger.info(f"Loaded data for {len(core_etf_data)} core ETFs")
            
            # Main backtest loop
            for i, rebalance_date in enumerate(rebalance_dates):
                try:
                    logger.info(f"Rebalancing {i+1}/{len(rebalance_dates)}: {rebalance_date.date()}")
                    
                    # Get stock universe and scores for this date
                    stock_scores = self._get_historical_scores(
                        rebalance_date, data_provider, scorer
                    )
                    
                    if not stock_scores:
                        logger.warning(f"No stock scores available for {rebalance_date.date()}")
                        continue
                        
                    # Select top stocks
                    selected_stocks = stock_scores[:self.num_satellite_stocks]
                    
                    # Calculate target allocation
                    target_allocation = self._calculate_target_allocation(
                        selected_stocks, core_etfs, portfolio_value
                    )
                    
                    # Get current prices
                    current_prices = self._get_prices_for_date(
                        rebalance_date, target_allocation.keys(), data_provider, core_etf_data
                    )
                    
                    if not current_prices:
                        logger.warning(f"No price data available for {rebalance_date.date()}")
                        continue
                        
                    # Execute rebalancing
                    new_holdings, new_cash, trades = self._execute_rebalancing(
                        holdings, cash, target_allocation, current_prices, rebalance_date
                    )
                    
                    # Update portfolio state
                    holdings = new_holdings
                    cash = new_cash
                    self.trades.extend(trades)
                    
                    # Calculate portfolio value
                    portfolio_value = cash + sum(
                        holdings.get(symbol, 0) * current_prices.get(symbol, 0)
                        for symbol in holdings
                    )
                    
                    # Record values
                    portfolio_values.append(portfolio_value)
                    
                    # Get benchmark value
                    if rebalance_date in benchmark_data.index:
                        benchmark_price = benchmark_data.loc[rebalance_date, 'Close']
                        benchmark_value = initial_capital * (benchmark_price / benchmark_data.iloc[0]['Close'])
                        benchmark_values.append(benchmark_value)
                    else:
                        benchmark_values.append(benchmark_values[-1] if benchmark_values else initial_capital)
                        
                    dates.append(rebalance_date)
                    
                except Exception as e:
                    logger.error(f"Error in rebalancing on {rebalance_date}: {e}")
                    continue
                    
            # Create portfolio history DataFrame
            self.portfolio_history = pd.DataFrame({
                'Date': dates,
                'Portfolio_Value': portfolio_values,
                'Benchmark_Value': benchmark_values
            }).set_index('Date')
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                self.portfolio_history, initial_capital
            )
            
            # Calculate trade statistics
            trade_stats = self._calculate_trade_statistics()
            
            # Combine results
            self.results = {
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
                'total_return': (portfolio_values[-1] / initial_capital - 1) if portfolio_values else 0,
                'num_rebalances': len(rebalance_dates),
                'num_trades': len(self.trades),
                **metrics,
                **trade_stats
            }
            
            logger.info(f"Backtest completed. Final value: ${self.results['final_value']:,.2f}")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def _generate_rebalance_dates(self, start_date: pd.Timestamp, 
                                end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """Generate rebalancing dates based on frequency.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of rebalancing dates
        """
        dates = []
        current_date = start_date
        
        if self.rebalance_frequency == 'monthly':
            # First trading day of each month
            while current_date <= end_date:
                # Move to first day of month
                first_of_month = current_date.replace(day=1)
                
                # Find first weekday (Monday=0, Sunday=6)
                while first_of_month.weekday() > 4:  # Skip weekends
                    first_of_month += timedelta(days=1)
                    
                if first_of_month <= end_date:
                    dates.append(first_of_month)
                    
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
                    
        elif self.rebalance_frequency == 'quarterly':
            # First trading day of each quarter
            quarters = [1, 4, 7, 10]  # January, April, July, October
            
            year = start_date.year
            while year <= end_date.year:
                for quarter_month in quarters:
                    quarter_date = pd.Timestamp(year, quarter_month, 1)
                    
                    if quarter_date < start_date:
                        continue
                    if quarter_date > end_date:
                        break
                        
                    # Find first weekday
                    while quarter_date.weekday() > 4:
                        quarter_date += timedelta(days=1)
                        
                    if quarter_date <= end_date:
                        dates.append(quarter_date)
                        
                year += 1
                
        return sorted(dates)
        
    def _get_benchmark_data(self, start_date: str, end_date: str, 
                          data_provider) -> pd.DataFrame:
        """Get benchmark (SPY) data for comparison.
        
        Args:
            start_date: Start date
            end_date: End date
            data_provider: Data provider instance
            
        Returns:
            Benchmark price data
        """
        try:
            benchmark_data = data_provider.get_historical_prices('SPY', start_date, end_date)
            
            if benchmark_data is None or benchmark_data.empty:
                # Fallback: create synthetic benchmark
                logger.warning("No benchmark data available, using synthetic data")
                dates = pd.date_range(start_date, end_date, freq='D')
                # Assume 8% annual return with 15% volatility
                daily_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), len(dates))
                prices = 100 * np.cumprod(1 + daily_returns)
                benchmark_data = pd.DataFrame({'Close': prices}, index=dates)
                
            return benchmark_data
            
        except Exception as e:
            logger.error(f"Error getting benchmark data: {e}")
            return pd.DataFrame()
            
    def _get_historical_scores(self, date: pd.Timestamp, data_provider, 
                             scorer) -> List[Dict]:
        """Get historical stock scores for a specific date.
        
        Args:
            date: Date to get scores for
            data_provider: Data provider instance
            scorer: Scorer instance
            
        Returns:
            List of scored stocks
        """
        try:
            # In a real implementation, this would use historical fundamental data
            # For backtesting, we'll use a simplified approach
            
            # Get a sample of stocks (in practice, this would be the full universe)
            sample_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
                'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'NFLX',
                'CRM', 'INTC', 'VZ', 'KO', 'PFE', 'T', 'XOM', 'CVX', 'WMT', 'BAC',
                'ABBV', 'TMO', 'COST', 'AVGO', 'ACN', 'DHR', 'TXN', 'NEE', 'LIN',
                'HON', 'QCOM', 'UPS', 'LOW', 'SBUX', 'MDT', 'IBM', 'AMT', 'SPGI'
            ]
            
            # Create mock fundamental data for backtesting
            scored_stocks = []
            
            for symbol in sample_stocks:
                try:
                    # Generate realistic but random scores for backtesting
                    np.random.seed(hash(symbol + str(date.date())) % 2**32)
                    
                    # Simulate fundamental metrics
                    roe = np.random.normal(0.15, 0.08)  # 15% average ROE
                    revenue_growth = np.random.normal(0.08, 0.12)  # 8% average growth
                    fcf_margin = np.random.normal(0.12, 0.06)  # 12% average FCF margin
                    peg_ratio = np.random.lognormal(0.5, 0.4)  # PEG around 1.6
                    gross_margin_stability = np.random.uniform(0.5, 0.95)
                    
                    # Apply hard filters (simplified)
                    market_cap = np.random.lognormal(10, 1.5)  # Billions
                    net_income = np.random.normal(1000, 500)  # Millions
                    debt_to_equity = np.random.lognormal(-0.5, 0.8)
                    
                    # Check hard filters
                    if (market_cap < 2 or  # $2B minimum
                        net_income <= 0 or  # Positive net income
                        revenue_growth <= 0.05 or  # >5% revenue growth
                        debt_to_equity >= 1.0):  # D/E < 1.0
                        continue
                        
                    # Calculate composite score
                    score_components = {
                        'roe_score': min(4, max(0, roe * 20)),  # Scale to 0-4
                        'growth_score': min(4, max(0, revenue_growth * 25)),
                        'fcf_score': min(4, max(0, fcf_margin * 20)),
                        'peg_score': min(4, max(0, 4 - peg_ratio)),  # Lower PEG = higher score
                        'stability_score': gross_margin_stability * 4
                    }
                    
                    # Weighted composite score
                    weights = self.config.get('scoring', {}).get('weights', {
                        'roe_weight': 0.25,
                        'growth_weight': 0.25,
                        'fcf_weight': 0.20,
                        'peg_weight': 0.20,
                        'stability_weight': 0.10
                    })
                    
                    composite_score = (
                        score_components['roe_score'] * weights.get('roe_weight', 0.25) +
                        score_components['growth_score'] * weights.get('growth_weight', 0.25) +
                        score_components['fcf_score'] * weights.get('fcf_weight', 0.20) +
                        score_components['peg_score'] * weights.get('peg_weight', 0.20) +
                        score_components['stability_score'] * weights.get('stability_weight', 0.10)
                    )
                    
                    scored_stocks.append({
                        'symbol': symbol,
                        'composite_score': composite_score,
                        'market_cap': market_cap,
                        'fundamentals': {
                            'roe': roe,
                            'revenue_growth': revenue_growth,
                            'fcf_margin': fcf_margin,
                            'peg_ratio': peg_ratio,
                            'gross_margin_stability': gross_margin_stability
                        }
                    })
                    
                except Exception as e:
                    logger.debug(f"Error scoring {symbol}: {e}")
                    continue
                    
            # Sort by composite score (descending)
            scored_stocks.sort(key=lambda x: x['composite_score'], reverse=True)
            
            logger.debug(f"Generated scores for {len(scored_stocks)} stocks on {date.date()}")
            return scored_stocks
            
        except Exception as e:
            logger.error(f"Error getting historical scores: {e}")
            return []
            
    def _calculate_target_allocation(self, selected_stocks: List[Dict], 
                                   core_etfs: List[Dict], 
                                   portfolio_value: float) -> Dict[str, float]:
        """Calculate target dollar allocation for each position.
        
        Args:
            selected_stocks: Selected satellite stocks
            core_etfs: Core ETF configuration
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary mapping symbols to target dollar amounts
        """
        allocation = {}
        
        # Core ETF allocation
        core_value = portfolio_value * self.core_allocation
        total_core_weight = sum(etf['weight'] for etf in core_etfs)
        
        for etf in core_etfs:
            symbol = etf['symbol']
            weight = etf['weight'] / total_core_weight  # Normalize weights
            allocation[symbol] = core_value * weight
            
        # Satellite stock allocation
        satellite_value = portfolio_value * self.satellite_allocation
        
        if selected_stocks:
            value_per_stock = satellite_value / len(selected_stocks)
            for stock in selected_stocks:
                allocation[stock['symbol']] = value_per_stock
                
        return allocation
        
    def _get_prices_for_date(self, date: pd.Timestamp, symbols: List[str], 
                           data_provider, core_etf_data: Dict) -> Dict[str, float]:
        """Get prices for symbols on a specific date.
        
        Args:
            date: Date to get prices for
            symbols: List of symbols
            data_provider: Data provider instance
            core_etf_data: Pre-loaded core ETF data
            
        Returns:
            Dictionary mapping symbols to prices
        """
        prices = {}
        
        for symbol in symbols:
            try:
                # Use pre-loaded ETF data if available
                if symbol in core_etf_data:
                    etf_data = core_etf_data[symbol]
                    # Find closest date
                    available_dates = etf_data.index
                    closest_date = min(available_dates, key=lambda x: abs((x - date).days))
                    
                    if abs((closest_date - date).days) <= 5:  # Within 5 days
                        prices[symbol] = etf_data.loc[closest_date, 'Close']
                        continue
                        
                # For stocks, generate synthetic prices for backtesting
                # In practice, this would fetch historical data
                np.random.seed(hash(symbol + str(date.date())) % 2**32)
                
                # Generate realistic stock price (between $10 and $500)
                base_price = np.random.lognormal(4, 1)  # Log-normal distribution
                prices[symbol] = max(10, min(500, base_price))
                
            except Exception as e:
                logger.debug(f"Error getting price for {symbol} on {date}: {e}")
                continue
                
        return prices
        
    def _execute_rebalancing(self, current_holdings: Dict[str, float], 
                           current_cash: float, target_allocation: Dict[str, float],
                           prices: Dict[str, float], date: pd.Timestamp) -> Tuple[Dict, float, List]:
        """Execute rebalancing trades.
        
        Args:
            current_holdings: Current holdings (symbol -> shares)
            current_cash: Current cash balance
            target_allocation: Target allocation (symbol -> dollar amount)
            prices: Current prices (symbol -> price)
            date: Trading date
            
        Returns:
            Tuple of (new_holdings, new_cash, trades)
        """
        new_holdings = current_holdings.copy()
        new_cash = current_cash
        trades = []
        
        # Calculate current position values
        current_values = {}
        for symbol, shares in current_holdings.items():
            if symbol in prices:
                current_values[symbol] = shares * prices[symbol]
            else:
                current_values[symbol] = 0
                
        # Calculate total portfolio value
        total_value = new_cash + sum(current_values.values())
        
        # Process each target position
        for symbol, target_value in target_allocation.items():
            if symbol not in prices:
                continue
                
            price = prices[symbol]
            current_value = current_values.get(symbol, 0)
            current_shares = new_holdings.get(symbol, 0)
            
            # Calculate target shares
            target_shares = target_value / price
            
            # Calculate trade size
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) > 0.01:  # Minimum trade threshold
                trade_value = abs(shares_diff * price)
                
                # Apply transaction costs
                transaction_cost = trade_value * self.transaction_cost
                slippage_cost = trade_value * self.slippage
                total_cost = transaction_cost + slippage_cost
                
                if shares_diff > 0:  # Buy
                    total_cost_with_fees = trade_value + total_cost
                    
                    if new_cash >= total_cost_with_fees:
                        new_holdings[symbol] = target_shares
                        new_cash -= total_cost_with_fees
                        
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'side': 'buy',
                            'shares': shares_diff,
                            'price': price,
                            'value': trade_value,
                            'fees': total_cost
                        })
                        
                else:  # Sell
                    new_holdings[symbol] = target_shares
                    proceeds = trade_value - total_cost
                    new_cash += proceeds
                    
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'side': 'sell',
                        'shares': abs(shares_diff),
                        'price': price,
                        'value': trade_value,
                        'fees': total_cost
                    })
                    
        # Remove zero positions
        new_holdings = {k: v for k, v in new_holdings.items() if v > 0.01}
        
        return new_holdings, new_cash, trades
        
    def _calculate_performance_metrics(self, portfolio_history: pd.DataFrame, 
                                     initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics.
        
        Args:
            portfolio_history: Portfolio value history
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary with performance metrics
        """
        if portfolio_history.empty:
            return {}
            
        try:
            # Calculate returns
            portfolio_returns = portfolio_history['Portfolio_Value'].pct_change().dropna()
            benchmark_returns = portfolio_history['Benchmark_Value'].pct_change().dropna()
            
            # Time period
            start_date = portfolio_history.index[0]
            end_date = portfolio_history.index[-1]
            years = (end_date - start_date).days / 365.25
            
            # Portfolio metrics
            final_value = portfolio_history['Portfolio_Value'].iloc[-1]
            total_return = (final_value / initial_capital) - 1
            cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility (annualized)
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = portfolio_returns - risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            # Maximum drawdown
            rolling_max = portfolio_history['Portfolio_Value'].expanding().max()
            drawdowns = (portfolio_history['Portfolio_Value'] - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Benchmark comparison
            benchmark_final = portfolio_history['Benchmark_Value'].iloc[-1]
            benchmark_total_return = (benchmark_final / initial_capital) - 1
            benchmark_cagr = (benchmark_final / initial_capital) ** (1 / years) - 1 if years > 0 else 0
            
            alpha = cagr - benchmark_cagr
            
            # Beta calculation
            if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            else:
                beta = 1
                
            # Information ratio
            tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (cagr - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Win rate
            positive_returns = portfolio_returns[portfolio_returns > 0]
            win_rate = len(positive_returns) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5) if len(portfolio_returns) > 0 else 0
            
            metrics = {
                'total_return': total_return,
                'cagr': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'win_rate': win_rate,
                'var_95': var_95,
                'benchmark_return': benchmark_total_return,
                'benchmark_cagr': benchmark_cagr,
                'years': years
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
            
    def _calculate_trade_statistics(self) -> Dict:
        """Calculate trade-related statistics.
        
        Returns:
            Dictionary with trade statistics
        """
        if not self.trades:
            return {}
            
        try:
            trades_df = pd.DataFrame(self.trades)
            
            # Basic trade stats
            total_trades = len(trades_df)
            buy_trades = len(trades_df[trades_df['side'] == 'buy'])
            sell_trades = len(trades_df[trades_df['side'] == 'sell'])
            
            # Trade values
            total_trade_value = trades_df['value'].sum()
            total_fees = trades_df['fees'].sum()
            
            # Average trade size
            avg_trade_value = trades_df['value'].mean()
            
            # Turnover calculation (annual)
            if len(self.portfolio_history) > 0:
                avg_portfolio_value = self.portfolio_history['Portfolio_Value'].mean()
                years = len(self.portfolio_history) / 12  # Assuming monthly rebalancing
                annual_turnover = (total_trade_value / 2) / avg_portfolio_value / years if years > 0 else 0
            else:
                annual_turnover = 0
                
            # Fee impact
            fee_percentage = total_fees / total_trade_value if total_trade_value > 0 else 0
            
            trade_stats = {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'total_trade_value': total_trade_value,
                'total_fees': total_fees,
                'avg_trade_value': avg_trade_value,
                'annual_turnover': annual_turnover,
                'fee_percentage': fee_percentage
            }
            
            return trade_stats
            
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return {}
            
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio value history.
        
        Returns:
            DataFrame with portfolio history
        """
        return self.portfolio_history.copy()
        
    def get_trades(self) -> List[Dict]:
        """Get list of all trades.
        
        Returns:
            List of trade dictionaries
        """
        return self.trades.copy()
        
    def save_results(self, output_dir: str = 'logs'):
        """Save backtest results to files.
        
        Args:
            output_dir: Output directory
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save portfolio history
            if not self.portfolio_history.empty:
                portfolio_file = f"{output_dir}/portfolio_history_{timestamp}.csv"
                self.portfolio_history.to_csv(portfolio_file)
                logger.info(f"Portfolio history saved to {portfolio_file}")
                
            # Save trades
            if self.trades:
                trades_file = f"{output_dir}/trades_{timestamp}.csv"
                pd.DataFrame(self.trades).to_csv(trades_file, index=False)
                logger.info(f"Trades saved to {trades_file}")
                
            # Save results summary
            if self.results:
                results_file = f"{output_dir}/backtest_results_{timestamp}.json"
                import json
                with open(results_file, 'w') as f:
                    json.dump(self.results, f, indent=2, default=str)
                logger.info(f"Results saved to {results_file}")
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")