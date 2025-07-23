# CNBC Contrarian Analysis System

This system analyzes CNBC articles to identify repetitive buy/sell patterns and provides contrarian trading signals based on the principle that media consensus often marks market extremes.

## How It Works

1. **Data Collection**: Fetches articles from multiple CNBC RSS feeds (Top News, Business, Investing, Markets)
2. **Sentiment Analysis**: Uses VADER sentiment analysis to score article sentiment
3. **Pattern Detection**: Identifies stocks with repetitive coverage and sentiment
4. **Contrarian Signals**: Generates opposite recommendations when media shows strong consensus
5. **Confidence Scoring**: Provides confidence levels based on repetition frequency

## API Endpoints

### `/cnbc-analysis` (POST)
Full CNBC analysis with custom parameters
```json
{
  "days_back": 7,
  "min_frequency": 3
}
```

### `/cnbc-quick-analysis` (GET)
Quick analysis with default parameters (3 days, min 2 mentions)

### `/cnbc-stock/{symbol}` (GET)
Analyze specific stock coverage (e.g., `/cnbc-stock/AAPL`)

## Chat Integration

The system automatically integrates with the chat interface when users mention:
- "cnbc"
- "contrarian" 
- "sentiment"
- "media"
- "news"
- "pattern"
- "opposite"

## Example Usage

```python
# Get contrarian recommendations
analysis = cnbc_analyzer.get_contrarian_recommendations(days_back=7, min_frequency=3)

# Results include:
# - buy_signals: Stocks to buy (contrarian to bearish media)
# - sell_signals: Stocks to sell (contrarian to bullish media)  
# - hold_signals: Stocks with mixed/neutral signals
# - confidence_score: 0-100% based on pattern strength
```

## Theory Behind Contrarian Analysis

1. **Media Saturation**: When media repeatedly covers a stock with the same sentiment, it often indicates market saturation
2. **Sentiment Extremes**: High repetition of bullish/bearish coverage often marks tops/bottoms
3. **Contrarian Opportunity**: Going against strong media consensus can be profitable
4. **Risk Management**: Higher confidence scores indicate stronger contrarian signals

## Dependencies

- requests: Web scraping
- beautifulsoup4: HTML/XML parsing
- vaderSentiment: Sentiment analysis
- yfinance: Stock symbol validation
- collections: Pattern analysis

## Testing

Run `test_cnbc.py` to verify RSS feed access and functionality.