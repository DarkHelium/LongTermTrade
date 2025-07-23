import requests
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import time
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@dataclass
class NewsArticle:
    title: str
    url: str
    published_date: datetime
    content: str
    sentiment_score: float
    mentioned_stocks: List[str]
    recommendation: str  # 'buy', 'sell', 'hold', 'neutral'
    source: str

@dataclass
class PatternAnalysis:
    stock_symbol: str
    repetitive_sentiment: str  # 'bullish', 'bearish', 'neutral'
    frequency: int
    confidence_score: float
    contrarian_signal: str  # 'buy', 'sell', 'hold'
    reasoning: str
    articles_analyzed: int
    time_period: str

class CNBCAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stock_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        self.buy_keywords = [
            'buy', 'bullish', 'upgrade', 'outperform', 'strong buy', 'overweight',
            'positive', 'rally', 'surge', 'soar', 'climb', 'gain', 'rise',
            'target price increase', 'price target raised', 'analyst upgrade',
            'recommend', 'attractive', 'opportunity', 'undervalued'
        ]
        self.sell_keywords = [
            'sell', 'bearish', 'downgrade', 'underperform', 'strong sell', 'underweight',
            'negative', 'decline', 'fall', 'drop', 'plunge', 'crash', 'tumble',
            'target price decrease', 'price target lowered', 'analyst downgrade',
            'avoid', 'overvalued', 'risk', 'concern', 'warning'
        ]
        
        # Headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
    def fetch_news_articles(self, days_back: int = 7) -> List[NewsArticle]:
        """Fetch recent financial news articles from multiple sources"""
        articles = []
        
        # Use multiple news sources for better coverage
        sources = [
            self._fetch_yahoo_finance_news,
            self._fetch_marketwatch_news,
            self._fetch_seeking_alpha_news,
            self._simulate_cnbc_patterns  # Fallback simulation for demo
        ]
        
        for source_func in sources:
            try:
                source_articles = source_func(days_back)
                articles.extend(source_articles)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"Error fetching from source: {e}")
                
        return articles
    
    def _fetch_yahoo_finance_news(self, days_back: int) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance"""
        articles = []
        try:
            # Yahoo Finance RSS feeds
            rss_urls = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://feeds.finance.yahoo.com/rss/2.0/topstories"
            ]
            
            for url in rss_urls:
                try:
                    response = requests.get(url, headers=self.headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'xml')
                        items = soup.find_all('item')
                        
                        for item in items[:10]:  # Limit to recent articles
                            title = item.title.text if item.title else ""
                            link = item.link.text if item.link else ""
                            description = item.description.text if item.description else ""
                            
                            # Create article object
                            article = self._create_article_from_text(
                                title, link, title + " " + description, "Yahoo Finance"
                            )
                            if article:
                                articles.append(article)
                                
                except Exception as e:
                    print(f"Error with Yahoo RSS {url}: {e}")
                    
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {e}")
            
        return articles
    
    def _fetch_marketwatch_news(self, days_back: int) -> List[NewsArticle]:
        """Fetch news from MarketWatch"""
        articles = []
        try:
            # MarketWatch RSS
            url = "http://feeds.marketwatch.com/marketwatch/topstories/"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                for item in items[:10]:
                    title = item.title.text if item.title else ""
                    link = item.link.text if item.link else ""
                    description = item.description.text if item.description else ""
                    
                    article = self._create_article_from_text(
                        title, link, title + " " + description, "MarketWatch"
                    )
                    if article:
                        articles.append(article)
                        
        except Exception as e:
            print(f"Error fetching MarketWatch news: {e}")
            
        return articles
    
    def _fetch_seeking_alpha_news(self, days_back: int) -> List[NewsArticle]:
        """Fetch news from Seeking Alpha"""
        articles = []
        try:
            # Seeking Alpha RSS
            url = "https://seekingalpha.com/api/sa/combined/A.xml"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                for item in items[:10]:
                    title = item.title.text if item.title else ""
                    link = item.link.text if item.link else ""
                    description = item.description.text if item.description else ""
                    
                    article = self._create_article_from_text(
                        title, link, title + " " + description, "Seeking Alpha"
                    )
                    if article:
                        articles.append(article)
                        
        except Exception as e:
            print(f"Error fetching Seeking Alpha news: {e}")
            
        return articles
    
    def _simulate_cnbc_patterns(self, days_back: int) -> List[NewsArticle]:
        """Simulate CNBC-style patterns for demonstration"""
        articles = []
        
        # Simulate repetitive patterns for popular stocks
        patterns = [
            {"stock": "AAPL", "sentiment": "bullish", "count": 4, "content": "Apple stock continues to show strong momentum with analyst upgrades and positive earnings outlook."},
            {"stock": "TSLA", "sentiment": "bearish", "count": 3, "content": "Tesla faces headwinds with production concerns and increased competition in EV market."},
            {"stock": "NVDA", "sentiment": "bullish", "count": 5, "content": "NVIDIA benefits from AI boom with strong demand for chips and data center growth."},
            {"stock": "META", "sentiment": "bearish", "count": 3, "content": "Meta struggles with metaverse investments and regulatory challenges affecting growth."},
            {"stock": "GOOGL", "sentiment": "bullish", "count": 4, "content": "Google shows resilience in search and cloud business despite economic headwinds."}
        ]
        
        for pattern in patterns:
            for i in range(pattern["count"]):
                title = f"{pattern['stock']} {pattern['sentiment']} outlook continues - Analysis"
                content = pattern["content"]
                
                article = self._create_article_from_text(
                    title, f"https://cnbc.com/simulated/{pattern['stock']}-{i}", 
                    content, "CNBC (Simulated)"
                )
                if article:
                    articles.append(article)
        
        return articles
    
    def _create_article_from_text(self, title: str, url: str, content: str, source: str) -> Optional[NewsArticle]:
        """Create NewsArticle object from text data"""
        try:
            # Extract mentioned stocks
            mentioned_stocks = self._extract_stock_symbols(title + " " + content)
            
            if not mentioned_stocks:  # Skip articles without stock mentions
                return None
            
            # Analyze sentiment
            sentiment_score = self._analyze_sentiment(title + " " + content)
            
            # Determine recommendation
            recommendation = self._determine_recommendation(title + " " + content)
            
            return NewsArticle(
                title=title,
                url=url,
                published_date=datetime.now(),
                content=content,
                sentiment_score=sentiment_score,
                mentioned_stocks=mentioned_stocks,
                recommendation=recommendation,
                source=source
            )
            
        except Exception as e:
            print(f"Error creating article: {e}")
            return None
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract potential stock symbols from text"""
        # Common stock symbols pattern
        potential_symbols = self.stock_pattern.findall(text.upper())
        
        # Filter out common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'NEW', 'NOW', 'OLD',
            'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE',
            'CEO', 'CFO', 'IPO', 'ETF', 'SEC', 'FDA', 'FTC', 'DOJ', 'GDP', 'CPI', 'USA', 'NYC',
            'CNBC', 'NYSE', 'NASDAQ', 'SPX', 'DOW', 'QQQ', 'SPY', 'RSS', 'XML', 'API', 'URL'
        }
        
        # Known stock symbols for validation
        known_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BABA', 'V',
            'JPM', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'CRM',
            'INTC', 'VZ', 'KO', 'PFE', 'T', 'XOM', 'BAC', 'ABBV', 'TMO', 'COST', 'AVGO'
        }
        
        valid_symbols = []
        for symbol in potential_symbols:
            if symbol not in false_positives and len(symbol) <= 5:
                if symbol in known_symbols:
                    valid_symbols.append(symbol)
                    
        return list(set(valid_symbols))
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using VADER"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores['compound']
    
    def _determine_recommendation(self, text: str) -> str:
        """Determine buy/sell/hold recommendation from text"""
        text_lower = text.lower()
        
        buy_score = sum(1 for keyword in self.buy_keywords if keyword in text_lower)
        sell_score = sum(1 for keyword in self.sell_keywords if keyword in text_lower)
        
        if buy_score > sell_score and buy_score > 0:
            return 'buy'
        elif sell_score > buy_score and sell_score > 0:
            return 'sell'
        elif buy_score == sell_score and buy_score > 0:
            return 'hold'
        else:
            return 'neutral'
    
    def analyze_patterns(self, articles: List[NewsArticle], min_frequency: int = 3) -> List[PatternAnalysis]:
        """Analyze repetitive patterns in news coverage"""
        stock_recommendations = defaultdict(list)
        stock_sentiments = defaultdict(list)
        stock_sources = defaultdict(set)
        
        # Group articles by stock
        for article in articles:
            for stock in article.mentioned_stocks:
                stock_recommendations[stock].append(article.recommendation)
                stock_sentiments[stock].append(article.sentiment_score)
                stock_sources[stock].add(article.source)
        
        patterns = []
        
        for stock, recommendations in stock_recommendations.items():
            if len(recommendations) < min_frequency:
                continue
                
            # Count recommendation types
            rec_counter = Counter(recommendations)
            most_common_rec, frequency = rec_counter.most_common(1)[0]
            
            # Calculate average sentiment
            avg_sentiment = sum(stock_sentiments[stock]) / len(stock_sentiments[stock])
            
            # Determine repetitive sentiment
            if avg_sentiment > 0.1:
                repetitive_sentiment = 'bullish'
            elif avg_sentiment < -0.1:
                repetitive_sentiment = 'bearish'
            else:
                repetitive_sentiment = 'neutral'
            
            # Generate contrarian signal
            contrarian_signal = self._generate_contrarian_signal(
                most_common_rec, repetitive_sentiment, frequency, len(recommendations)
            )
            
            # Calculate confidence score (higher for more sources and repetition)
            source_diversity = len(stock_sources[stock])
            repetition_ratio = frequency / len(recommendations)
            confidence_score = min((repetition_ratio * 0.7 + (source_diversity / 4) * 0.3) * 100, 100)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                stock, most_common_rec, repetitive_sentiment, frequency, len(recommendations), source_diversity
            )
            
            pattern = PatternAnalysis(
                stock_symbol=stock,
                repetitive_sentiment=repetitive_sentiment,
                frequency=frequency,
                confidence_score=confidence_score,
                contrarian_signal=contrarian_signal,
                reasoning=reasoning,
                articles_analyzed=len(recommendations),
                time_period=f"Recent analysis across {source_diversity} sources"
            )
            
            patterns.append(pattern)
        
        # Sort by confidence score
        return sorted(patterns, key=lambda x: x.confidence_score, reverse=True)
    
    def _generate_contrarian_signal(self, most_common_rec: str, sentiment: str, frequency: int, total: int) -> str:
        """Generate contrarian trading signal"""
        repetition_ratio = frequency / total
        
        # Strong contrarian signal if high repetition
        if repetition_ratio >= 0.7:
            if most_common_rec == 'buy' or sentiment == 'bullish':
                return 'sell'
            elif most_common_rec == 'sell' or sentiment == 'bearish':
                return 'buy'
        
        # Moderate contrarian signal
        elif repetition_ratio >= 0.5:
            if most_common_rec == 'buy' or sentiment == 'bullish':
                return 'hold'
            elif most_common_rec == 'sell' or sentiment == 'bearish':
                return 'hold'
        
        return 'hold'
    
    def _generate_reasoning(self, stock: str, rec: str, sentiment: str, frequency: int, total: int, sources: int) -> str:
        """Generate human-readable reasoning for the contrarian signal"""
        repetition_ratio = frequency / total
        
        reasoning = f"Media analysis shows {stock} mentioned in {total} articles across {sources} sources with {frequency} showing {rec} sentiment ({sentiment}). "
        reasoning += f"Repetition rate: {repetition_ratio:.1%}. "
        
        if repetition_ratio >= 0.7:
            reasoning += "High repetition suggests potential media saturation and market consensus. "
            reasoning += "Contrarian approach recommended as strong media consensus often marks market extremes."
        elif repetition_ratio >= 0.5:
            reasoning += "Moderate repetition detected. Consider cautious contrarian positioning."
        else:
            reasoning += "Low repetition - no strong contrarian signal detected."
            
        return reasoning
    
    def get_contrarian_recommendations(self, days_back: int = 7, min_frequency: int = 3) -> Dict:
        """Main method to get contrarian trading recommendations"""
        print(f"Fetching financial news from the last {days_back} days...")
        articles = self.fetch_news_articles(days_back)
        
        print(f"Analyzing {len(articles)} articles for patterns...")
        patterns = self.analyze_patterns(articles, min_frequency)
        
        # Categorize recommendations
        buy_signals = [p for p in patterns if p.contrarian_signal == 'buy']
        sell_signals = [p for p in patterns if p.contrarian_signal == 'sell']
        hold_signals = [p for p in patterns if p.contrarian_signal == 'hold']
        
        return {
            'analysis_date': datetime.now().isoformat(),
            'articles_analyzed': len(articles),
            'patterns_found': len(patterns),
            'buy_signals': [self._pattern_to_dict(p) for p in buy_signals],
            'sell_signals': [self._pattern_to_dict(p) for p in sell_signals],
            'hold_signals': [self._pattern_to_dict(p) for p in hold_signals],
            'summary': {
                'total_stocks_analyzed': len(patterns),
                'strong_buy_signals': len([p for p in buy_signals if p.confidence_score > 70]),
                'strong_sell_signals': len([p for p in sell_signals if p.confidence_score > 70]),
                'time_period': f"Last {days_back} days",
                'sources_used': list(set(article.source for article in articles))
            }
        }
    
    def _pattern_to_dict(self, pattern: PatternAnalysis) -> Dict:
        """Convert PatternAnalysis to dictionary"""
        return {
            'stock_symbol': pattern.stock_symbol,
            'repetitive_sentiment': pattern.repetitive_sentiment,
            'frequency': pattern.frequency,
            'confidence_score': round(pattern.confidence_score, 2),
            'contrarian_signal': pattern.contrarian_signal,
            'reasoning': pattern.reasoning,
            'articles_analyzed': pattern.articles_analyzed,
            'time_period': pattern.time_period
        }