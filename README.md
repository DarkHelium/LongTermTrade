# Long-Term Investment Bot

A sophisticated Python-based investment bot designed for long-term wealth building through systematic stock selection, portfolio management, and risk control. The bot implements a fundamental analysis approach combined with modern portfolio theory to compound capital over 5-10 year horizons.

## Strategy Overview

The bot follows a disciplined investment approach:

- **70% Core Holdings**: Broad-market ETFs (VTI, VXUS) for market exposure
- **30% Satellite Holdings**: High-quality individual stocks selected via fundamental analysis
- **Monthly Rebalancing**: Systematic review and rebalancing on first trading day of each month
- **Fundamental Scorecard**: Multi-factor scoring system for stock selection
- **Risk Management**: Built-in sell rules, concentration limits, and position sizing
- **Optional Sentiment Overlay**: Contrarian sentiment analysis for enhanced timing

## Key Features

### 🎯 **Automated Monthly Workflow**
- Fetches updated fundamentals for all US stocks
- Scores stocks using proprietary fundamental scorecard
- Selects top-ranked stocks passing hard filters
- Rebalances portfolio to target allocations
- Executes trades via Alpaca API

### 📊 **Fundamental Scorecard**
- **Hard Filters**: Market cap >$2B, positive earnings, revenue growth >5%, debt/equity <1.0
- **Scoring Metrics**: 5-year ROE, revenue CAGR, FCF margin, PEG ratio, gross margin stability
- **Composite Score**: Weighted combination of all metrics (0-4 scale)

### 🛡️ **Risk Management**
- **Broken Thesis**: Sell stocks with negative revenue growth + low scores
- **Overvaluation**: Trim positions with extreme valuations (PE >90th percentile + PEG >2.5)
- **Concentration**: Limit individual positions to 20% of portfolio
- **Better Replacement**: Swap holdings when significantly better opportunities arise

### 📈 **Backtesting & Analytics**
- Vectorized backtester covering 20+ years of data
- Performance metrics: CAGR, Sharpe ratio, max drawdown, alpha, beta
- Trade analysis and turnover statistics
- Risk-adjusted return calculations

### 🔧 **Configuration & Extensibility**
- YAML-based configuration for all parameters
- Modular architecture for easy customization
- Command-line interface for different modes
- Comprehensive logging and monitoring

## Installation

### Prerequisites
- Python 3.11 or higher
- Alpaca brokerage account (paper trading enabled by default)
- Free API keys for data sources (Alpha Vantage, Finnhub)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DayTradeBot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   # Alpaca API (Paper Trading)
   ALPACA_API_KEY=your_paper_api_key
   ALPACA_SECRET_KEY=your_paper_secret_key
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   
   # Data Sources
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   FINNHUB_API_KEY=your_finnhub_key
   
   # Optional: Notifications
   SLACK_WEBHOOK_URL=your_slack_webhook
   EMAIL_SMTP_SERVER=smtp.gmail.com
   EMAIL_USERNAME=your_email
   EMAIL_PASSWORD=your_app_password
   ```

5. **Test the installation**
   ```bash
   python -m pytest tests/ -v
   ```

## Usage

### Command Line Interface

The bot supports multiple operation modes:

```bash
# Run backtest (default: 2004-2024)
python main.py --backtest

# Run backtest with custom date range
python main.py --backtest --start-date 2020-01-01 --end-date 2023-12-31

# Paper trading mode (safe for testing)
python main.py --paper

# Live trading mode (real money - use with caution)
python main.py --live

# Force immediate rebalancing
python main.py --paper --rebalance-now

# Display portfolio summary
python main.py --paper --summary

# Check market status
python main.py --market-status

# Run with custom config file
python main.py --config custom_config.yml --paper
```

### Configuration

The bot is configured via `config.yml`. Key sections include:

```yaml
portfolio:
  core_allocation: 0.70        # 70% in ETFs
  satellite_allocation: 0.30   # 30% in individual stocks
  core_etfs:
    VTI: 0.60                  # 60% of core in VTI
    VXUS: 0.40                 # 40% of core in VXUS
  max_satellite_stocks: 20     # Max number of individual stocks

scoring:
  weights:
    roe_5yr: 0.25              # 5-year average ROE weight
    revenue_cagr_5yr: 0.25     # Revenue CAGR weight
    fcf_margin: 0.20           # Free cash flow margin weight
    peg_ratio: 0.15            # PEG ratio weight (lower is better)
    gross_margin_stability: 0.15 # Gross margin consistency weight

  hard_filters:
    min_market_cap: 2000000000 # $2B minimum market cap
    min_net_income: 0          # Positive earnings required
    min_revenue_growth: 0.05   # 5% minimum revenue growth
    max_debt_to_equity: 1.0    # Maximum debt-to-equity ratio
```

### Automated Scheduling

For production use, set up automated execution:

**Windows Task Scheduler:**
```bash
# Create a batch file (run_bot.bat)
cd C:\path\to\DayTradeBot
.venv\Scripts\activate
python main.py --paper

# Schedule to run on first trading day of each month
```

**Linux/Mac Cron:**
```bash
# Add to crontab (crontab -e)
# Run at 9:35 AM ET on first weekday of each month
35 9 1-7 * * [ "$(date +\%u)" -le 5 ] && cd /path/to/DayTradeBot && .venv/bin/activate && python main.py --paper
```

## Architecture

The bot follows a modular design pattern:

```
DayTradeBot/
├── main.py              # Main orchestrator and CLI
├── config.yml           # Configuration file
├── data.py              # Data fetching and processing
├── scoring.py           # Fundamental analysis and scoring
├── portfolio.py         # Portfolio management and allocation
├── broker.py            # Alpaca API integration
├── sentiment.py         # Sentiment analysis (optional)
├── backtest.py          # Backtesting engine
├── tests/               # Unit tests
│   ├── test_scoring.py
│   ├── test_portfolio.py
│   └── test_data.py
├── logs/                # Log files and backtest results
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Key Components

- **DataProvider**: Fetches fundamental and price data from free sources
- **FundamentalScorer**: Implements scoring algorithm and sell conditions
- **PortfolioManager**: Handles allocation, rebalancing, and position sizing
- **AlpacaBroker**: Manages order execution and account information
- **SentimentAnalyzer**: Optional contrarian sentiment overlay
- **LongTermBacktester**: Vectorized backtesting engine

## Performance Metrics

The bot tracks comprehensive performance metrics:

- **Returns**: CAGR, total return, annual returns
- **Risk**: Volatility, max drawdown, downside deviation
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Benchmark**: Alpha, beta, tracking error vs. market
- **Trading**: Turnover, win rate, average holding period

## Data Sources

The bot uses free data sources to keep costs low:

- **Alpha Vantage**: Fundamental data, company overviews
- **Yahoo Finance**: Historical prices, basic fundamentals
- **Finnhub**: Industry peers, additional fundamental data
- **Reddit/News APIs**: Sentiment data (optional)

## Risk Considerations

⚠️ **Important Disclaimers:**

- This software is for educational and research purposes
- Past performance does not guarantee future results
- All investments carry risk of loss
- Start with paper trading to understand the system
- Never invest more than you can afford to lose
- Consider consulting with a financial advisor

### Risk Management Features

- **Paper Trading Default**: Safe testing environment
- **Position Limits**: Maximum 20% in any single stock
- **Diversification**: Forced diversification across 20+ holdings
- **Systematic Selling**: Automated sell rules for risk control
- **Cash Buffer**: Maintains small cash position for opportunities

## Customization

### Adding New Scoring Factors

```python
# In scoring.py, add new scoring method
def _score_new_factor(self, value):
    """Score a new fundamental factor."""
    if value >= 0.20: return 4
    elif value >= 0.15: return 3
    elif value >= 0.10: return 2
    elif value >= 0.05: return 1
    else: return 0

# Update calculate_scores method to include new factor
```

### Custom Sell Rules

```python
# In scoring.py, modify check_sell_conditions
def check_sell_conditions(self, stock_data, universe_data, industry_data, portfolio_weight):
    # Add custom sell logic
    custom_sell = self._check_custom_condition(stock_data)
    
    return {
        'broken_thesis': broken_thesis,
        'overvaluation': overvaluation,
        'concentration': concentration,
        'custom_rule': custom_sell  # New rule
    }
```

### Alternative Data Sources

```python
# In data.py, add new data provider
class CustomDataProvider:
    def get_fundamentals(self, symbol):
        # Implement custom data fetching
        pass
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Increase `rate_limit_delay` in config
   - Use multiple API keys if available
   - Consider caching data locally

2. **Missing Data**
   - Check API key validity
   - Verify symbol exists and is actively traded
   - Review data source status

3. **Order Execution Failures**
   - Ensure sufficient buying power
   - Check market hours
   - Verify Alpaca account status

4. **Performance Issues**
   - Reduce universe size for faster processing
   - Use SSD storage for better I/O
   - Consider running on cloud instance

### Logging

Comprehensive logging is available in the `logs/` directory:

- `bot.log`: General application logs
- `trades.log`: Trade execution details
- `backtest_YYYYMMDD.log`: Backtest results
- `errors.log`: Error tracking

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes `ruff` and `black` formatting
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install ruff black pytest-cov

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Format code
black .
ruff check . --fix
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Review the documentation
- Check existing issues for solutions
- Consider the troubleshooting section

---

**Remember**: This is a long-term investment strategy. Be patient, stay disciplined, and let compound growth work in your favor over time.