"""Scoring module for fundamental analysis and stock ranking.

This module implements the fundamental scorecard with hard filters
and composite scoring based on financial metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalScorer:
    """Fundamental scoring engine for stock selection."""
    
    def __init__(self, config: Dict):
        """Initialize scorer with configuration.
        
        Args:
            config: Configuration dictionary with scoring weights and filters
        """
        self.config = config
        self.hard_filters = config.get('hard_filters', {})
        self.scoring_weights = config.get('scoring_weights', {})
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.scoring_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Scoring weights sum to {weight_sum:.3f}, not 1.0")
            
    def apply_hard_filters(self, fundamentals_data: List[Dict]) -> List[Dict]:
        """Apply hard filters to screen stocks.
        
        Args:
            fundamentals_data: List of fundamental data dictionaries
            
        Returns:
            Filtered list of stocks that pass all hard filters
        """
        filtered_stocks = []
        
        for stock_data in fundamentals_data:
            if self._passes_hard_filters(stock_data):
                filtered_stocks.append(stock_data)
                
        logger.info(f"Hard filters: {len(filtered_stocks)}/{len(fundamentals_data)} stocks passed")
        return filtered_stocks
        
    def _passes_hard_filters(self, stock_data: Dict) -> bool:
        """Check if a stock passes all hard filters.
        
        Args:
            stock_data: Stock fundamental data
            
        Returns:
            True if stock passes all filters
        """
        try:
            # Market cap filter
            min_market_cap = self.hard_filters.get('min_market_cap', 2e9)
            if stock_data.get('market_cap', 0) < min_market_cap:
                return False
                
            # TTM net income filter (positive)
            min_net_income = self.hard_filters.get('min_ttm_net_income', 0)
            if stock_data.get('ttm_net_income', 0) < min_net_income:
                return False
                
            # YoY revenue growth filter
            min_revenue_growth = self.hard_filters.get('min_revenue_growth_yoy', 0.05)
            if stock_data.get('revenue_growth', 0) < min_revenue_growth:
                return False
                
            # Debt-to-equity filter
            max_debt_to_equity = self.hard_filters.get('max_debt_to_equity', 1.0)
            if stock_data.get('debt_to_equity', 0) > max_debt_to_equity:
                return False
                
            # Additional filters can be added here
            
            return True
            
        except Exception as e:
            logger.warning(f"Error applying hard filters to {stock_data.get('symbol', 'unknown')}: {e}")
            return False
            
    def calculate_composite_scores(self, filtered_stocks: List[Dict]) -> List[Dict]:
        """Calculate composite scores for filtered stocks.
        
        Args:
            filtered_stocks: List of stocks that passed hard filters
            
        Returns:
            List of stocks with composite scores added
        """
        if not filtered_stocks:
            return []
            
        # Calculate individual metric scores
        scored_stocks = []
        for stock_data in filtered_stocks:
            try:
                scores = self._calculate_individual_scores(stock_data, filtered_stocks)
                stock_data['individual_scores'] = scores
                stock_data['composite_score'] = self._calculate_weighted_score(scores)
                scored_stocks.append(stock_data)
            except Exception as e:
                logger.warning(f"Error scoring {stock_data.get('symbol', 'unknown')}: {e}")
                continue
                
        # Sort by composite score (descending)
        scored_stocks.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        logger.info(f"Calculated composite scores for {len(scored_stocks)} stocks")
        return scored_stocks
        
    def _calculate_individual_scores(self, stock_data: Dict, universe: List[Dict]) -> Dict:
        """Calculate individual metric scores (0-4 scale).
        
        Args:
            stock_data: Individual stock data
            universe: All stocks for percentile calculations
            
        Returns:
            Dictionary of individual metric scores
        """
        scores = {}
        
        # 5-year average ROE score
        scores['roe_5yr'] = self._score_roe(stock_data, universe)
        
        # 5-year revenue CAGR score
        scores['revenue_cagr_5yr'] = self._score_revenue_cagr(stock_data, universe)
        
        # Free cash flow margin score
        scores['fcf_margin'] = self._score_fcf_margin(stock_data, universe)
        
        # Forward PEG ratio score (inverted - lower is better)
        scores['peg_ratio'] = self._score_peg_ratio(stock_data, universe)
        
        # Gross margin stability score (moat proxy)
        scores['gross_margin_stability'] = self._score_gross_margin_stability(stock_data, universe)
        
        return scores
        
    def _score_roe(self, stock_data: Dict, universe: List[Dict]) -> float:
        """Score return on equity (0-4 scale)."""
        roe = stock_data.get('return_on_equity', 0)
        
        if roe <= 0:
            return 0.0
            
        # Get ROE distribution from universe
        roe_values = [s.get('return_on_equity', 0) for s in universe if s.get('return_on_equity', 0) > 0]
        
        if not roe_values:
            return 2.0  # Neutral score
            
        percentile = self._calculate_percentile(roe, roe_values)
        return self._percentile_to_score(percentile)
        
    def _score_revenue_cagr(self, stock_data: Dict, universe: List[Dict]) -> float:
        """Score 5-year revenue CAGR (0-4 scale)."""
        cagr = stock_data.get('revenue_cagr_5yr', 0)
        
        if cagr <= 0:
            return 0.0
            
        # Get CAGR distribution from universe
        cagr_values = [s.get('revenue_cagr_5yr', 0) for s in universe if s.get('revenue_cagr_5yr', 0) > 0]
        
        if not cagr_values:
            return 2.0  # Neutral score
            
        percentile = self._calculate_percentile(cagr, cagr_values)
        return self._percentile_to_score(percentile)
        
    def _score_fcf_margin(self, stock_data: Dict, universe: List[Dict]) -> float:
        """Score free cash flow margin (0-4 scale)."""
        fcf_margin = stock_data.get('fcf_margin', 0)
        
        if fcf_margin <= 0:
            return 0.0
            
        # Get FCF margin distribution from universe
        fcf_values = [s.get('fcf_margin', 0) for s in universe if s.get('fcf_margin', 0) > 0]
        
        if not fcf_values:
            return 2.0  # Neutral score
            
        percentile = self._calculate_percentile(fcf_margin, fcf_values)
        return self._percentile_to_score(percentile)
        
    def _score_peg_ratio(self, stock_data: Dict, universe: List[Dict]) -> float:
        """Score PEG ratio (0-4 scale, inverted - lower is better)."""
        peg = stock_data.get('peg_ratio', 0)
        
        if peg <= 0 or peg > 10:  # Invalid or extreme PEG
            return 0.0
            
        # Get PEG distribution from universe
        peg_values = [s.get('peg_ratio', 0) for s in universe 
                     if 0 < s.get('peg_ratio', 0) <= 10]
        
        if not peg_values:
            return 2.0  # Neutral score
            
        # Invert percentile for PEG (lower is better)
        percentile = self._calculate_percentile(peg, peg_values)
        inverted_percentile = 100 - percentile
        return self._percentile_to_score(inverted_percentile)
        
    def _score_gross_margin_stability(self, stock_data: Dict, universe: List[Dict]) -> float:
        """Score gross margin stability as moat proxy (0-4 scale)."""
        # This is a simplified implementation
        # In practice, you'd calculate std dev of gross margins over 5 years
        gross_margin = stock_data.get('gross_margins', 0)
        
        if gross_margin <= 0:
            return 0.0
            
        # Higher gross margin generally indicates better moat
        # This is a proxy until we have historical margin data
        margin_values = [s.get('gross_margins', 0) for s in universe 
                        if s.get('gross_margins', 0) > 0]
        
        if not margin_values:
            return 2.0  # Neutral score
            
        percentile = self._calculate_percentile(gross_margin, margin_values)
        return self._percentile_to_score(percentile)
        
    def _calculate_percentile(self, value: float, distribution: List[float]) -> float:
        """Calculate percentile rank of value in distribution."""
        if not distribution:
            return 50.0
            
        distribution = sorted(distribution)
        n = len(distribution)
        
        # Find position in sorted list
        position = 0
        for i, val in enumerate(distribution):
            if value <= val:
                position = i
                break
        else:
            position = n
            
        return (position / n) * 100
        
    def _percentile_to_score(self, percentile: float) -> float:
        """Convert percentile to 0-4 score."""
        if percentile >= 90:
            return 4.0
        elif percentile >= 75:
            return 3.0
        elif percentile >= 50:
            return 2.0
        elif percentile >= 25:
            return 1.0
        else:
            return 0.0
            
    def _calculate_weighted_score(self, individual_scores: Dict) -> float:
        """Calculate weighted composite score.
        
        Args:
            individual_scores: Dictionary of individual metric scores
            
        Returns:
            Weighted composite score
        """
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.scoring_weights.items():
            if metric in individual_scores:
                total_score += individual_scores[metric] * weight
                total_weight += weight
                
        if total_weight == 0:
            return 0.0
            
        return total_score / total_weight
        
    def select_top_stocks(self, scored_stocks: List[Dict], max_stocks: int) -> List[Dict]:
        """Select top N stocks based on composite score.
        
        Args:
            scored_stocks: List of stocks with composite scores
            max_stocks: Maximum number of stocks to select
            
        Returns:
            Top N stocks
        """
        if not scored_stocks:
            return []
            
        # Already sorted by composite score in calculate_composite_scores
        selected = scored_stocks[:max_stocks]
        
        logger.info(f"Selected top {len(selected)} stocks from {len(scored_stocks)} candidates")
        
        # Log top selections
        for i, stock in enumerate(selected[:10], 1):
            symbol = stock.get('symbol', 'Unknown')
            score = stock.get('composite_score', 0)
            logger.info(f"  {i}. {symbol}: {score:.2f}")
            
        return selected
        
    def check_sell_conditions(self, current_holdings: List[Dict], 
                            universe_scores: List[Dict]) -> Dict[str, str]:
        """Check sell/trim conditions for current holdings.
        
        Args:
            current_holdings: List of current portfolio holdings
            universe_scores: List of all scored stocks in universe
            
        Returns:
            Dictionary mapping symbols to sell reasons
        """
        sell_decisions = {}
        
        # Create lookup for universe scores
        universe_lookup = {stock['symbol']: stock for stock in universe_scores}
        
        for holding in current_holdings:
            symbol = holding.get('symbol', '')
            current_data = universe_lookup.get(symbol)
            
            if not current_data:
                continue
                
            # Check broken thesis
            if self._check_broken_thesis(current_data, universe_scores):
                sell_decisions[symbol] = 'broken_thesis'
                continue
                
            # Check extreme overvaluation
            if self._check_extreme_overvaluation(current_data, universe_scores):
                sell_decisions[symbol] = 'overvaluation'
                continue
                
            # Check concentration cap
            portfolio_weight = holding.get('weight', 0)
            max_weight = self.config.get('sell_rules', {}).get('concentration', {}).get('max_position_weight', 0.20)
            if portfolio_weight > max_weight:
                sell_decisions[symbol] = 'concentration'
                continue
                
            # Check better replacement
            if self._check_better_replacement(current_data, universe_scores, current_holdings):
                sell_decisions[symbol] = 'replacement'
                
        return sell_decisions
        
    def _check_broken_thesis(self, stock_data: Dict, universe: List[Dict]) -> bool:
        """Check if stock has broken thesis (negative revenue growth + bottom 30%)."""
        revenue_growth = stock_data.get('revenue_growth', 0)
        composite_score = stock_data.get('composite_score', 0)
        
        # Negative revenue growth
        if revenue_growth >= 0:
            return False
            
        # Bottom 30% of universe
        scores = [s.get('composite_score', 0) for s in universe]
        if not scores:
            return False
            
        bottom_30_threshold = np.percentile(scores, 30)
        return composite_score <= bottom_30_threshold
        
    def _check_extreme_overvaluation(self, stock_data: Dict, universe: List[Dict]) -> bool:
        """Check if stock is extremely overvalued."""
        forward_pe = stock_data.get('forward_pe', 0)
        peg_ratio = stock_data.get('peg_ratio', 0)
        industry = stock_data.get('industry', '')
        
        if forward_pe <= 0 or peg_ratio <= 0:
            return False
            
        # Get industry peers
        industry_peers = [s for s in universe if s.get('industry', '') == industry]
        if len(industry_peers) < 5:  # Not enough peers
            return False
            
        # Check if P/E is in 90th percentile of industry
        industry_pes = [s.get('forward_pe', 0) for s in industry_peers if s.get('forward_pe', 0) > 0]
        if not industry_pes:
            return False
            
        pe_90th_percentile = np.percentile(industry_pes, 90)
        max_peg = self.config.get('sell_rules', {}).get('overvaluation', {}).get('max_peg_ratio', 2.5)
        
        return forward_pe > pe_90th_percentile and peg_ratio > max_peg
        
    def _check_better_replacement(self, stock_data: Dict, universe: List[Dict], 
                                current_holdings: List[Dict]) -> bool:
        """Check if there's a significantly better replacement available."""
        current_score = stock_data.get('composite_score', 0)
        threshold = self.config.get('sell_rules', {}).get('replacement', {}).get('score_improvement_threshold', 25)
        
        # Get symbols of current holdings
        held_symbols = {h.get('symbol', '') for h in current_holdings}
        
        # Find best non-held stock
        best_available_score = 0
        for candidate in universe:
            if candidate.get('symbol', '') not in held_symbols:
                candidate_score = candidate.get('composite_score', 0)
                best_available_score = max(best_available_score, candidate_score)
                
        return best_available_score - current_score >= threshold
        
    def get_scoring_summary(self, scored_stocks: List[Dict]) -> Dict:
        """Generate summary statistics for scoring results.
        
        Args:
            scored_stocks: List of scored stocks
            
        Returns:
            Summary statistics dictionary
        """
        if not scored_stocks:
            return {}
            
        scores = [s.get('composite_score', 0) for s in scored_stocks]
        
        summary = {
            'total_stocks': len(scored_stocks),
            'score_mean': np.mean(scores),
            'score_median': np.median(scores),
            'score_std': np.std(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'top_10_symbols': [s.get('symbol', '') for s in scored_stocks[:10]],
            'scoring_weights': self.scoring_weights,
            'hard_filters': self.hard_filters
        }
        
        return summary