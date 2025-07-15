# Long-Term Investment System powered by Fin-R1

A comprehensive investment analysis system that leverages the Fin-R1 financial AI model to make intelligent long-term investment decisions. This system combines real-time financial data collection with advanced AI analysis to provide actionable investment recommendations.

## 🚀 Features

- **AI-Powered Analysis**: Uses Fin-R1 model for sophisticated financial reasoning
- **Real-Time Data**: Integrates with Yahoo Finance and other financial APIs
- **Company Analysis**: Deep dive into individual companies with comprehensive metrics
- **Portfolio Management**: Analyze and optimize existing portfolios
- **Investment Screening**: Find opportunities based on custom criteria
- **Sector Analysis**: Understand sector trends and opportunities
- **Risk Assessment**: Comprehensive risk evaluation and warnings
- **ESG Integration**: Environmental, Social, and Governance considerations

## 📋 Prerequisites

1. **Fin-R1 Model**: Download and set up the Fin-R1 model locally
2. **Python 3.8+**: Ensure you have Python 3.8 or higher installed
3. **vLLM Server**: For serving the Fin-R1 model

## 🛠️ Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd LongTermTrade
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .myenv
   source .myenv/bin/activate  # On Windows: .myenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the Fin-R1 model**:
   ```bash
   # Start the vLLM server with Fin-R1
   vllm serve /Users/anavmadan/Desktop/LongTermTrade/Fin-R1 \
     --host 0.0.0.0 \
     --port 8000 \
     --api-key your-api-key
   ```

## 🎯 Quick Start

### 1. Create Sample Files
```bash
python long_term_trader.py create-samples
```
This creates `sample_portfolio.json` and `sample_criteria.json` files.

### 2. Analyze a Single Company
```bash
# Basic analysis
python long_term_trader.py analyze AAPL

# Analysis with custom investment amount
python long_term_trader.py analyze AAPL --amount 25000
```

### 3. Analyze Your Portfolio
```bash
python long_term_trader.py portfolio sample_portfolio.json
```

### 4. Screen for Investment Opportunities
```bash
python long_term_trader.py screen sample_criteria.json --max-results 15
```

### 5. Sector Analysis
```bash
python long_term_trader.py sector Technology
python long_term_trader.py sector Healthcare
```

### 6. Compare Companies
```bash
python long_term_trader.py compare AAPL MSFT GOOGL AMZN
```

## 📁 File Formats

### Portfolio File (JSON)
```json
{
  "AAPL": 15000,
  "MSFT": 12000,
  "GOOGL": 10000,
  "AMZN": 8000,
  "TSLA": 5000
}
```

### Screening Criteria File (JSON)
```json
{
  "min_market_cap": 1000000000,
  "max_pe": 30,
  "min_roe": 0.15,
  "max_debt_to_equity": 1.0,
  "sectors": ["Technology", "Healthcare", "Consumer Discretionary"]
}
```

## 🏗️ System Architecture

### Core Components

1. **FinR1Client** (`fin_r1_client.py`)
   - Interfaces with the Fin-R1 model
   - Handles AI-powered financial analysis
   - Provides structured prompts for different analysis types

2. **DataCollector** (`data_collector.py`)
   - Collects real-time financial data
   - Integrates with Yahoo Finance API
   - Provides market conditions and company fundamentals

3. **InvestmentAnalyzer** (`investment_analyzer.py`)
   - Main analysis engine
   - Combines data collection with AI analysis
   - Generates structured investment recommendations

4. **LongTermTrader** (`long_term_trader.py`)
   - Command-line interface
   - Orchestrates analysis workflows
   - Provides formatted output

### Data Flow
```
[Financial APIs] → [DataCollector] → [InvestmentAnalyzer] → [Fin-R1 Model] → [Recommendations]
```

## 📊 Analysis Types

### Company Analysis
- Financial health assessment
- Valuation analysis
- Growth potential evaluation
- Risk factor identification
- ESG considerations
- Competitive positioning

### Portfolio Analysis
- Diversification scoring
- Risk level assessment
- Expected return calculation
- Sector allocation analysis
- Rebalancing recommendations
- Performance optimization

### Screening Analysis
- Custom criteria filtering
- Fundamental analysis
- Ranking by confidence scores
- Risk-adjusted recommendations

### Sector Analysis
- Industry trends and outlook
- Competitive landscape
- Growth drivers and risks
- Top investment picks
- Valuation comparisons

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
FIN_R1_API_URL=http://localhost:8000/v1
FIN_R1_API_KEY=your-api-key
NEWS_API_KEY=your-news-api-key  # Optional
ALPHA_VANTAGE_KEY=your-av-key    # Optional
```

### Logging
Logs are written to `long_term_trader.log` and console output.

## 📈 Example Output

### Company Analysis
```
🔍 Analyzing AAPL for long-term investment...
============================================================

🏢 Company: Apple Inc. (AAPL)
📊 Recommendation: BUY
🎯 Confidence Score: 87%
⏰ Time Horizon: Long
💰 Target Price: $195.50

📈 Scores:
- Financial Health: 92/100
- Valuation: 78/100
- Growth Potential: 85/100
- Market Position: 95/100

💡 Key Growth Drivers:
- Services revenue expansion
- iPhone upgrade cycles
- Emerging market penetration

⚠️  Risk Factors:
- Regulatory pressures
- Supply chain dependencies
- Market saturation
```

## 🛡️ Risk Management

The system provides comprehensive risk assessment:
- **Market Risk**: Volatility and correlation analysis
- **Company Risk**: Financial health and business model risks
- **Sector Risk**: Industry-specific challenges
- **Regulatory Risk**: Compliance and policy changes
- **ESG Risk**: Environmental and governance factors

## 🔄 Continuous Monitoring

For production use, consider:
- Regular portfolio rebalancing
- Market condition monitoring
- News sentiment analysis
- Performance tracking
- Risk limit enforcement

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📝 Logging and Monitoring

- All analysis results are logged
- Performance metrics tracked
- Error handling and recovery
- Audit trail for decisions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## ⚠️ Disclaimer

**This system is for educational and research purposes only. It does not constitute financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the logs in `long_term_trader.log`
2. Ensure Fin-R1 server is running
3. Verify API connectivity
4. Review sample files format

## 🔮 Future Enhancements

- Real-time portfolio tracking
- Advanced backtesting capabilities
- Integration with brokers for live trading
- Enhanced ESG scoring
- Alternative data sources
- Machine learning model improvements
- Mobile app interface
- Social sentiment analysis