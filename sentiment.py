"""Sentiment analysis module for contrarian sentiment overlay.

This module provides sentiment analysis capabilities using Reddit r/stocks
and news headlines to implement a contrarian sentiment strategy.
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analyzer for stocks using Reddit and news sources."""
    
    def __init__(self, config: Dict):
        """Initialize sentiment analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        sentiment_config = config.get('sentiment', {})
        
        self.enabled = sentiment_config.get('enabled', False)
        self.reddit_enabled = sentiment_config.get('reddit_enabled', True)
        self.news_enabled = sentiment_config.get('news_enabled', True)
        
        # Sentiment thresholds
        self.exuberance_threshold = sentiment_config.get('exuberance_threshold', 0.9)  # 90th percentile
        self.consecutive_days_threshold = sentiment_config.get('consecutive_days_threshold', 10)
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Rate limiting
        self.last_reddit_request = 0
        self.reddit_rate_limit = 2  # seconds between requests
        
        # Cache for sentiment data
        self.sentiment_cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
        logger.info(f"Sentiment analyzer initialized - Enabled: {self.enabled}")
        
    def is_enabled(self) -> bool:
        """Check if sentiment analysis is enabled.
        
        Returns:
            True if enabled
        """
        return self.enabled
        
    def get_reddit_sentiment(self, symbol: str, limit: int = 100) -> Dict:
        """Get sentiment from Reddit r/stocks for a specific symbol.
        
        Args:
            symbol: Stock symbol
            limit: Number of posts to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        if not self.reddit_enabled:
            return {'sentiment_score': 0, 'post_count': 0, 'confidence': 0}
            
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_reddit_request < self.reddit_rate_limit:
                time.sleep(self.reddit_rate_limit - (current_time - self.last_reddit_request))
                
            # Check cache first
            cache_key = f"reddit_{symbol}_{limit}"
            if cache_key in self.sentiment_cache:
                cached_data, timestamp = self.sentiment_cache[cache_key]
                if current_time - timestamp < self.cache_duration:
                    return cached_data
                    
            # Reddit API endpoint (using public JSON API)
            url = f"https://www.reddit.com/r/stocks/search.json"
            params = {
                'q': f"${symbol} OR {symbol}",
                'restrict_sr': 'true',
                'sort': 'new',
                'limit': limit,
                't': 'week'  # Last week
            }
            
            headers = {
                'User-Agent': 'LongTermBot/1.0 (Investment Research)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            self.last_reddit_request = time.time()
            
            if response.status_code != 200:
                logger.warning(f"Reddit API request failed: {response.status_code}")
                return {'sentiment_score': 0, 'post_count': 0, 'confidence': 0}
                
            data = response.json()
            posts = data.get('data', {}).get('children', [])
            
            if not posts:
                return {'sentiment_score': 0, 'post_count': 0, 'confidence': 0}
                
            # Analyze sentiment of posts
            sentiments = []
            valid_posts = 0
            
            for post in posts:
                post_data = post.get('data', {})
                
                # Combine title and selftext
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                text = f"{title} {selftext}".strip()
                
                if len(text) < 10:  # Skip very short posts
                    continue
                    
                # Check if post is actually about the symbol
                if not self._is_relevant_post(text, symbol):
                    continue
                    
                # Analyze sentiment
                sentiment = self.vader.polarity_scores(text)
                
                # Weight by post score (upvotes)
                score = max(1, post_data.get('score', 1))
                weight = min(10, max(1, score))  # Cap weight at 10x
                
                sentiments.append({
                    'compound': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu'],
                    'weight': weight,
                    'score': score,
                    'created_utc': post_data.get('created_utc', 0)
                })
                
                valid_posts += 1
                
            if not sentiments:
                result = {'sentiment_score': 0, 'post_count': 0, 'confidence': 0}
            else:
                # Calculate weighted average sentiment
                total_weight = sum(s['weight'] for s in sentiments)
                weighted_sentiment = sum(s['compound'] * s['weight'] for s in sentiments) / total_weight
                
                # Calculate confidence based on number of posts and agreement
                sentiment_values = [s['compound'] for s in sentiments]
                sentiment_std = pd.Series(sentiment_values).std()
                confidence = min(1.0, valid_posts / 20) * (1 - min(1.0, sentiment_std))
                
                result = {
                    'sentiment_score': weighted_sentiment,
                    'post_count': valid_posts,
                    'confidence': confidence,
                    'raw_sentiments': sentiment_values,
                    'avg_score': sum(s['score'] for s in sentiments) / len(sentiments)
                }
                
            # Cache result
            self.sentiment_cache[cache_key] = (result, current_time)
            
            logger.info(f"Reddit sentiment for {symbol}: {result['sentiment_score']:.3f} ({valid_posts} posts)")
            return result
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
            return {'sentiment_score': 0, 'post_count': 0, 'confidence': 0}
            
    def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """Get sentiment from news headlines for a specific symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment metrics
        """
        if not self.news_enabled:
            return {'sentiment_score': 0, 'article_count': 0, 'confidence': 0}
            
        try:
            # Check cache first
            cache_key = f"news_{symbol}_{days}"
            current_time = time.time()
            if cache_key in self.sentiment_cache:
                cached_data, timestamp = self.sentiment_cache[cache_key]
                if current_time - timestamp < self.cache_duration:
                    return cached_data
                    
            # Use multiple free news sources
            headlines = []
            
            # Try NewsAPI (free tier)
            headlines.extend(self._get_newsapi_headlines(symbol, days))
            
            # Try Alpha Vantage news (if available)
            headlines.extend(self._get_alphavantage_news(symbol, days))
            
            # Try Yahoo Finance RSS (backup)
            headlines.extend(self._get_yahoo_news(symbol, days))
            
            if not headlines:
                result = {'sentiment_score': 0, 'article_count': 0, 'confidence': 0}
            else:
                # Analyze sentiment of headlines
                sentiments = []
                
                for headline in headlines:
                    if len(headline) < 10:  # Skip very short headlines
                        continue
                        
                    sentiment = self.vader.polarity_scores(headline)
                    sentiments.append(sentiment['compound'])
                    
                if not sentiments:
                    result = {'sentiment_score': 0, 'article_count': 0, 'confidence': 0}
                else:
                    # Calculate average sentiment
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    
                    # Calculate confidence based on number of articles and agreement
                    sentiment_std = pd.Series(sentiments).std()
                    confidence = min(1.0, len(sentiments) / 10) * (1 - min(1.0, sentiment_std))
                    
                    result = {
                        'sentiment_score': avg_sentiment,
                        'article_count': len(sentiments),
                        'confidence': confidence,
                        'raw_sentiments': sentiments
                    }
                    
            # Cache result
            self.sentiment_cache[cache_key] = (result, current_time)
            
            logger.info(f"News sentiment for {symbol}: {result['sentiment_score']:.3f} ({result['article_count']} articles)")
            return result
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {'sentiment_score': 0, 'article_count': 0, 'confidence': 0}
            
    def get_combined_sentiment(self, symbol: str) -> Dict:
        """Get combined sentiment from Reddit and news sources.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with combined sentiment metrics
        """
        if not self.enabled:
            return {
                'sentiment_score': 0,
                'confidence': 0,
                'reddit_sentiment': 0,
                'news_sentiment': 0,
                'data_sources': 0
            }
            
        try:
            reddit_data = self.get_reddit_sentiment(symbol)
            news_data = self.get_news_sentiment(symbol)
            
            # Combine sentiments with weights
            reddit_weight = 0.6 if reddit_data['post_count'] > 0 else 0
            news_weight = 0.4 if news_data['article_count'] > 0 else 0
            
            # Normalize weights
            total_weight = reddit_weight + news_weight
            if total_weight == 0:
                combined_sentiment = 0
                combined_confidence = 0
            else:
                reddit_weight /= total_weight
                news_weight /= total_weight
                
                combined_sentiment = (
                    reddit_data['sentiment_score'] * reddit_weight +
                    news_data['sentiment_score'] * news_weight
                )
                
                combined_confidence = (
                    reddit_data['confidence'] * reddit_weight +
                    news_data['confidence'] * news_weight
                )
                
            data_sources = (1 if reddit_data['post_count'] > 0 else 0) + (1 if news_data['article_count'] > 0 else 0)
            
            result = {
                'sentiment_score': combined_sentiment,
                'confidence': combined_confidence,
                'reddit_sentiment': reddit_data['sentiment_score'],
                'news_sentiment': news_data['sentiment_score'],
                'reddit_posts': reddit_data['post_count'],
                'news_articles': news_data['article_count'],
                'data_sources': data_sources
            }
            
            logger.info(f"Combined sentiment for {symbol}: {combined_sentiment:.3f} (confidence: {combined_confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error getting combined sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0,
                'confidence': 0,
                'reddit_sentiment': 0,
                'news_sentiment': 0,
                'data_sources': 0
            }
            
    def check_exuberance_condition(self, symbol: str, sentiment_history: List[Dict]) -> bool:
        """Check if a stock meets the exuberance condition for accelerated selling.
        
        Args:
            symbol: Stock symbol
            sentiment_history: List of historical sentiment data
            
        Returns:
            True if exuberance condition is met
        """
        if not self.enabled or len(sentiment_history) < self.consecutive_days_threshold:
            return False
            
        try:
            # Get recent sentiment scores
            recent_scores = []
            for entry in sentiment_history[-self.consecutive_days_threshold:]:
                if entry.get('symbol') == symbol and entry.get('confidence', 0) > 0.3:
                    recent_scores.append(entry.get('sentiment_score', 0))
                    
            if len(recent_scores) < self.consecutive_days_threshold:
                return False
                
            # Check if all recent scores are in top 10% (exuberance threshold)
            exuberant_count = sum(1 for score in recent_scores if score > self.exuberance_threshold)
            
            is_exuberant = exuberant_count >= self.consecutive_days_threshold
            
            if is_exuberant:
                logger.warning(f"Exuberance condition met for {symbol}: {exuberant_count}/{self.consecutive_days_threshold} days")
                
            return is_exuberant
            
        except Exception as e:
            logger.error(f"Error checking exuberance condition for {symbol}: {e}")
            return False
            
    def _is_relevant_post(self, text: str, symbol: str) -> bool:
        """Check if a post is relevant to the given symbol.
        
        Args:
            text: Post text
            symbol: Stock symbol
            
        Returns:
            True if relevant
        """
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        # Check for symbol mentions
        patterns = [
            f"\\${symbol_lower}\\b",  # $AAPL
            f"\\b{symbol_lower}\\b",   # AAPL
            f"\\b{symbol_lower}\\s",   # AAPL followed by space
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False
        
    def _get_newsapi_headlines(self, symbol: str, days: int) -> List[str]:
        """Get headlines from NewsAPI (free tier).
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of headlines
        """
        headlines = []
        
        try:
            # NewsAPI free tier (requires API key)
            api_key = self.config.get('newsapi_key')
            if not api_key:
                return headlines
                
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{symbol}" OR "${symbol}"',
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                for article in articles:
                    title = article.get('title', '')
                    if title and len(title) > 10:
                        headlines.append(title)
                        
        except Exception as e:
            logger.debug(f"NewsAPI request failed: {e}")
            
        return headlines
        
    def _get_alphavantage_news(self, symbol: str, days: int) -> List[str]:
        """Get headlines from Alpha Vantage news.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of headlines
        """
        headlines = []
        
        try:
            # Alpha Vantage news (requires API key)
            api_key = self.config.get('alpha_vantage_key')
            if not api_key:
                return headlines
                
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': api_key,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                feed = data.get('feed', [])
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for article in feed:
                    # Check date
                    time_published = article.get('time_published', '')
                    if time_published:
                        try:
                            article_date = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                            if article_date < cutoff_date:
                                continue
                        except:
                            continue
                            
                    title = article.get('title', '')
                    if title and len(title) > 10:
                        headlines.append(title)
                        
        except Exception as e:
            logger.debug(f"Alpha Vantage news request failed: {e}")
            
        return headlines
        
    def _get_yahoo_news(self, symbol: str, days: int) -> List[str]:
        """Get headlines from Yahoo Finance RSS (backup method).
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of headlines
        """
        headlines = []
        
        try:
            # Yahoo Finance RSS feed
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Simple XML parsing for RSS
                import xml.etree.ElementTree as ET
                
                root = ET.fromstring(response.content)
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for item in root.findall('.//item'):
                    title_elem = item.find('title')
                    pub_date_elem = item.find('pubDate')
                    
                    if title_elem is not None and title_elem.text:
                        title = title_elem.text.strip()
                        
                        # Check date if available
                        if pub_date_elem is not None and pub_date_elem.text:
                            try:
                                # Parse RFC 2822 date format
                                from email.utils import parsedate_to_datetime
                                article_date = parsedate_to_datetime(pub_date_elem.text)
                                if article_date.replace(tzinfo=None) < cutoff_date:
                                    continue
                            except:
                                pass  # Include if date parsing fails
                                
                        if len(title) > 10:
                            headlines.append(title)
                            
        except Exception as e:
            logger.debug(f"Yahoo Finance RSS request failed: {e}")
            
        return headlines
        
    def get_sentiment_summary(self, symbols: List[str]) -> Dict:
        """Get sentiment summary for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with sentiment summary
        """
        if not self.enabled:
            return {'enabled': False, 'symbols_analyzed': 0}
            
        summary = {
            'enabled': True,
            'symbols_analyzed': 0,
            'high_sentiment_symbols': [],
            'low_sentiment_symbols': [],
            'exuberant_symbols': [],
            'avg_sentiment': 0,
            'avg_confidence': 0
        }
        
        sentiments = []
        confidences = []
        
        for symbol in symbols:
            try:
                sentiment_data = self.get_combined_sentiment(symbol)
                
                if sentiment_data['data_sources'] > 0:
                    summary['symbols_analyzed'] += 1
                    
                    sentiment_score = sentiment_data['sentiment_score']
                    confidence = sentiment_data['confidence']
                    
                    sentiments.append(sentiment_score)
                    confidences.append(confidence)
                    
                    # Categorize by sentiment
                    if sentiment_score > 0.5:
                        summary['high_sentiment_symbols'].append({
                            'symbol': symbol,
                            'sentiment': sentiment_score,
                            'confidence': confidence
                        })
                    elif sentiment_score < -0.5:
                        summary['low_sentiment_symbols'].append({
                            'symbol': symbol,
                            'sentiment': sentiment_score,
                            'confidence': confidence
                        })
                        
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                continue
                
        # Calculate averages
        if sentiments:
            summary['avg_sentiment'] = sum(sentiments) / len(sentiments)
            summary['avg_confidence'] = sum(confidences) / len(confidences)
            
        logger.info(f"Sentiment summary: {summary['symbols_analyzed']} symbols analyzed")
        return summary