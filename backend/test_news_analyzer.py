#!/usr/bin/env python3
"""
Test script for the updated news analyzer system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnbc_analyzer import CNBCAnalyzer
import json

def test_news_sources():
    """Test fetching from multiple news sources"""
    print("Testing News Analyzer with Multiple Sources")
    print("=" * 50)
    
    analyzer = CNBCAnalyzer()
    
    # Test individual source methods
    print("\n1. Testing Yahoo Finance RSS...")
    try:
        yahoo_articles = analyzer._fetch_yahoo_finance_news(7)
        print(f"   ✓ Yahoo Finance: {len(yahoo_articles)} articles")
        if yahoo_articles:
            print(f"   Sample: {yahoo_articles[0].title[:80]}...")
    except Exception as e:
        print(f"   ✗ Yahoo Finance error: {e}")
    
    print("\n2. Testing MarketWatch RSS...")
    try:
        mw_articles = analyzer._fetch_marketwatch_news(7)
        print(f"   ✓ MarketWatch: {len(mw_articles)} articles")
        if mw_articles:
            print(f"   Sample: {mw_articles[0].title[:80]}...")
    except Exception as e:
        print(f"   ✗ MarketWatch error: {e}")
    
    print("\n3. Testing Seeking Alpha RSS...")
    try:
        sa_articles = analyzer._fetch_seeking_alpha_news(7)
        print(f"   ✓ Seeking Alpha: {len(sa_articles)} articles")
        if sa_articles:
            print(f"   Sample: {sa_articles[0].title[:80]}...")
    except Exception as e:
        print(f"   ✗ Seeking Alpha error: {e}")
    
    print("\n4. Testing Simulated CNBC Patterns...")
    try:
        sim_articles = analyzer._simulate_cnbc_patterns(7)
        print(f"   ✓ Simulated: {len(sim_articles)} articles")
        if sim_articles:
            print(f"   Sample: {sim_articles[0].title[:80]}...")
            print(f"   Stocks mentioned: {sim_articles[0].mentioned_stocks}")
    except Exception as e:
        print(f"   ✗ Simulation error: {e}")

def test_full_analysis():
    """Test the complete contrarian analysis"""
    print("\n" + "=" * 50)
    print("Testing Full Contrarian Analysis")
    print("=" * 50)
    
    analyzer = CNBCAnalyzer()
    
    try:
        # Get contrarian recommendations
        results = analyzer.get_contrarian_recommendations(days_back=7, min_frequency=2)
        
        print(f"\nAnalysis Results:")
        print(f"- Articles analyzed: {results['articles_analyzed']}")
        print(f"- Patterns found: {results['patterns_found']}")
        print(f"- Sources used: {', '.join(results['summary']['sources_used'])}")
        
        print(f"\nSignal Summary:")
        print(f"- Buy signals: {len(results['buy_signals'])}")
        print(f"- Sell signals: {len(results['sell_signals'])}")
        print(f"- Hold signals: {len(results['hold_signals'])}")
        
        # Show detailed buy signals
        if results['buy_signals']:
            print(f"\n🟢 CONTRARIAN BUY SIGNALS:")
            for signal in results['buy_signals'][:3]:  # Show top 3
                print(f"   {signal['stock_symbol']}: {signal['contrarian_signal'].upper()}")
                print(f"   Confidence: {signal['confidence_score']}%")
                print(f"   Reasoning: {signal['reasoning'][:100]}...")
                print()
        
        # Show detailed sell signals
        if results['sell_signals']:
            print(f"\n🔴 CONTRARIAN SELL SIGNALS:")
            for signal in results['sell_signals'][:3]:  # Show top 3
                print(f"   {signal['stock_symbol']}: {signal['contrarian_signal'].upper()}")
                print(f"   Confidence: {signal['confidence_score']}%")
                print(f"   Reasoning: {signal['reasoning'][:100]}...")
                print()
        
        # Save results to file for inspection
        with open('news_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Full results saved to 'news_analysis_results.json'")
        
    except Exception as e:
        print(f"Error in full analysis: {e}")
        import traceback
        traceback.print_exc()

def test_stock_extraction():
    """Test stock symbol extraction"""
    print("\n" + "=" * 50)
    print("Testing Stock Symbol Extraction")
    print("=" * 50)
    
    analyzer = CNBCAnalyzer()
    
    test_texts = [
        "Apple (AAPL) stock surges on strong earnings",
        "Tesla TSLA faces production challenges",
        "NVIDIA NVDA benefits from AI boom",
        "Microsoft and Google compete in cloud",
        "The market is volatile today with SPY down"
    ]
    
    for text in test_texts:
        stocks = analyzer._extract_stock_symbols(text)
        print(f"Text: {text}")
        print(f"Extracted stocks: {stocks}")
        print()

if __name__ == "__main__":
    print("🚀 Starting News Analyzer Tests")
    
    # Run all tests
    test_stock_extraction()
    test_news_sources()
    test_full_analysis()
    
    print("\n✅ Testing completed!")