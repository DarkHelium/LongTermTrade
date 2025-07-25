# Warren Buffett Stock Analysis Agent

An AI-powered stock analysis system that applies Warren Buffett's investment principles with an integrated **Financial Knowledge Graph** for structured memory and explainable reasoning.

## 🧠 New: Financial Knowledge Graph

This system now includes a sophisticated **words-only** financial knowledge graph that provides:

- **Structured Memory**: Store and retrieve financial facts in natural language
- **Explainable Reasoning**: Six-step reasoning ritual for transparent decision-making  
- **Fact Extraction**: Automatically extract facts from financial documents and news
- **Natural Language Queries**: Query the knowledge base using plain English
- **Temporal Tracking**: Track how financial relationships change over time
- **Confidence Scoring**: Every fact includes confidence levels and provenance

### Knowledge Graph Features

1. **Financial Ontology**: Canonical phrases for companies, metrics, relationships, and events
2. **Fact Storage**: Plain text files organized by entity for easy auditing
3. **LLM-Powered Extraction**: Convert raw text into structured facts
4. **Reasoning Engine**: Six-step process for answering complex financial queries
5. **Quality Control**: Confidence scoring and source tracking for all facts

## 🏗️ Architecture

The backend follows a modular architecture inspired by the `trae-agent` repository structure:

```
backend/
├── agent/                  # Core AI agent implementation
│   ├── __init__.py
│   └── warren_buffett_agent.py
├── core/                   # Core configuration and utilities
│   ├── __init__.py
│   └── config.py
├── prompts/                # System prompts and templates
│   ├── __init__.py
│   └── system_prompts.py
├── services/               # External service integrations
│   ├── __init__.py
│   └── llm_service.py
├── tools/                  # Analysis and data tools
│   ├── __init__.py
│   ├── analysis_tool.py
│   ├── market_data_tool.py
│   └── search_tool.py
├── api.py                  # FastAPI application
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## 🚀 Features

### Core Analysis Capabilities
- **Fundamental Analysis**: Deep dive into company financials using Buffett's criteria
- **Business Quality Assessment**: Evaluate competitive moats and management quality
- **Valuation Analysis**: Intrinsic value calculation with margin of safety
- **Risk Assessment**: Comprehensive risk evaluation framework
- **Investment Recommendations**: Clear buy/hold/sell recommendations with reasoning

### Knowledge Graph Capabilities
- **Text Ingestion**: Process 10-K filings, news articles, and research reports
- **Fact Extraction**: Automatically identify and store financial relationships
- **Natural Language Queries**: Ask questions like "What companies have high ROE?"
- **Entity Profiles**: Get comprehensive profiles of companies and investors
- **Enhanced Analysis**: Combine traditional analysis with knowledge graph insights

### Search and Screening
- **Stock Search**: Find stocks by symbol or company name
- **Quality Screening**: Filter stocks based on Buffett's quality criteria
- **Undervalued Stock Discovery**: Identify potentially undervalued opportunities
- **Dividend Stock Analysis**: Find attractive dividend-paying stocks

### AI-Powered Chat
- **Investment Guidance**: Chat with an AI that thinks like Warren Buffett
- **Educational Content**: Learn about value investing principles
- **Market Analysis**: Get insights on market conditions and opportunities

## 🛠️ Setup

### 1. Environment Setup

Copy the environment template:
```bash
cp .env.example .env
```

Configure your API keys in `.env`:
```env
# Required: Market Data
FINNHUB_API_KEY=your_finnhub_api_key

# Required: AI Services (choose one)
OPENAI_API_KEY=your_openai_api_key
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: Additional Data Sources
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret

# Configuration
LLM_PROVIDER=openai  # or 'anthropic'
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### Stock Analysis
- `POST /analyze` - Comprehensive stock analysis
- `POST /analyze/enhanced` - Enhanced analysis with knowledge graph insights
- `GET /popular` - Get popular stocks for analysis
- `POST /search` - Search stocks by symbol or name

### Screening & Filtering
- `POST /screen` - Screen quality stocks
- `GET /undervalued` - Find undervalued stocks
- `GET /dividends` - Find dividend stocks

### Knowledge Graph
- `POST /knowledge/ingest` - Ingest financial text and extract facts
- `POST /knowledge/query` - Query knowledge graph with natural language
- `GET /knowledge/entity/{entity}` - Get entity profile and facts
- `GET /knowledge/search` - Search for entities in knowledge graph
- `POST /knowledge/fact` - Manually add facts to knowledge graph
- `GET /knowledge/stats` - Get knowledge graph statistics
- `GET /knowledge/ontology` - Get financial ontology information

### AI Chat
- `POST /chat` - Chat with Warren Buffett agent

### System
- `GET /health` - Health check
- `GET /config` - Current configuration

## 🧠 Warren Buffett's Investment Principles

The agent implements these core principles:

### Business Quality
- **High ROE**: Consistent returns on equity (>15%)
- **Strong Margins**: Healthy profit margins
- **Low Debt**: Conservative debt levels
- **Predictable Earnings**: Consistent growth patterns

### Valuation
- **Margin of Safety**: Buy below intrinsic value
- **Reasonable P/E**: Avoid overvalued stocks
- **Long-term Value**: Focus on sustainable competitive advantages

### Investment Approach
- **Quality over Quantity**: Few, high-quality investments
- **Long-term Horizon**: Hold for years, not months
- **Understand the Business**: Invest in comprehensible companies
- **Management Quality**: Strong, shareholder-friendly leadership

## 🔧 Configuration

### LLM Providers
- **OpenAI**: GPT-4 for analysis and chat
- **Anthropic**: Claude-3 for analysis and chat

### Market Data Sources
- **Finnhub**: Primary real-time data source
- **Alpha Vantage**: Fundamental data backup
- **Alpaca**: Trading integration (optional)
- **Yahoo Finance**: Additional data source

### Caching & Performance
- **Memory Caching**: Fast repeated requests
- **Disk Caching**: Persistent data storage
- **Rate Limiting**: Respect API limits
- **Async Processing**: Non-blocking operations

## 📊 Example Usage

### Basic Stock Analysis
```python
import requests

# Analyze Apple stock
response = requests.post("http://localhost:8000/analyze", json={
    "symbol": "AAPL",
    "include_fundamentals": True,
    "include_valuation": True,
    "include_recommendation": True
})

analysis = response.json()
print(f"Recommendation: {analysis['analysis']['recommendation']}")
```

### Enhanced Analysis with Knowledge Graph
```python
# Enhanced analysis combining traditional metrics with knowledge graph insights
response = requests.post("http://localhost:8000/analyze/enhanced", json={
    "symbol": "AAPL"
})

enhanced_analysis = response.json()
print(f"Traditional Analysis: {enhanced_analysis['traditional_analysis']}")
print(f"Knowledge Graph Insights: {enhanced_analysis['knowledge_graph_insights']}")
```

### Knowledge Graph Operations
```python
# Ingest financial text
response = requests.post("http://localhost:8000/knowledge/ingest", json={
    "text": "Apple reported revenue of $394.3 billion in 2022, up 8% from 2021.",
    "source": "Apple 10-K 2022"
})

# Query the knowledge graph
response = requests.post("http://localhost:8000/knowledge/query", json={
    "query": "What was Apple's revenue in 2022?"
})

answer = response.json()
print(f"Answer: {answer['answer']}")
print(f"Reasoning: {answer['reasoning_steps']}")

# Get entity profile
response = requests.get("http://localhost:8000/knowledge/entity/Apple")
profile = response.json()
print(f"Facts about Apple: {len(profile['facts'])}")
```

### Chat with Warren Buffett AI
```python
# Chat about investment strategy
response = requests.post("http://localhost:8000/chat", json={
    "message": "What should I look for when analyzing a technology stock?"
})

chat_response = response.json()
print(f"Warren Buffett: {chat_response['response']}")
```

### Stock Screening
```python
# Find undervalued stocks
response = requests.get("http://localhost:8000/undervalued?max_pe=15&min_margin_safety=20")
undervalued = response.json()

for stock in undervalued['stocks']:
    print(f"{stock['symbol']}: P/E {stock['pe_ratio']}, Margin of Safety: {stock['margin_of_safety']}%")
```

## 🚨 Important Notes

### API Keys Required
- **Finnhub API**: Free tier available at [finnhub.io](https://finnhub.io)
- **OpenAI/Anthropic**: Paid APIs for AI features

### Rate Limits
- Respects all API provider rate limits
- Built-in caching to minimize API calls
- Configurable request throttling

### Data Accuracy
- Real-time market data subject to provider delays
- Analysis based on available fundamental data
- Not financial advice - for educational purposes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with all API provider terms of service.

## 🙏 Acknowledgments

- Warren Buffett for timeless investment wisdom
- Finnhub for reliable market data
- OpenAI/Anthropic for powerful AI capabilities
- The `trae-agent` project for architectural inspiration
