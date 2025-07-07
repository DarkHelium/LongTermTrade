"""Unit tests for the scoring module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from scoring import FundamentalScorer


class TestFundamentalScorer:
    """Test cases for FundamentalScorer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'scoring': {
                'weights': {
                    'roe_5yr': 0.25,
                    'revenue_cagr_5yr': 0.25,
                    'fcf_margin': 0.20,
                    'peg_ratio': 0.15,
                    'gross_margin_stability': 0.15
                },
                'hard_filters': {
                    'min_market_cap': 2e9,
                    'min_net_income': 0,
                    'min_revenue_growth': 0.05,
                    'max_debt_to_equity': 1.0
                },
                'sell_conditions': {
                    'broken_thesis_percentile': 0.30,
                    'overvaluation_pe_percentile': 0.90,
                    'overvaluation_peg_threshold': 2.5,
                    'concentration_threshold': 0.20,
                    'concentration_target': 0.15,
                    'replacement_score_threshold': 25
                }
            }
        }
        self.scorer = FundamentalScorer(self.config)

    def test_initialization(self):
        """Test scorer initialization."""
        assert self.scorer.weights == self.config['scoring']['weights']
        assert self.scorer.hard_filters == self.config['scoring']['hard_filters']
        assert self.scorer.sell_conditions == self.config['scoring']['sell_conditions']

    def test_apply_hard_filters_pass(self):
        """Test hard filters with stocks that should pass."""
        data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'market_cap': [3e12, 2.5e12, 1.5e12],
            'net_income_ttm': [1e11, 8e10, 6e10],
            'revenue_growth_yoy': [0.08, 0.12, 0.15],
            'debt_to_equity': [0.5, 0.3, 0.2]
        })
        
        filtered = self.scorer.apply_hard_filters(data)
        assert len(filtered) == 3
        assert list(filtered['symbol']) == ['AAPL', 'MSFT', 'GOOGL']

    def test_apply_hard_filters_fail(self):
        """Test hard filters with stocks that should fail."""
        data = pd.DataFrame({
            'symbol': ['SMALL', 'LOSS', 'NOGROWTH', 'DEBT'],
            'market_cap': [1e9, 3e12, 3e12, 3e12],  # SMALL fails market cap
            'net_income_ttm': [1e11, -1e10, 1e11, 1e11],  # LOSS fails net income
            'revenue_growth_yoy': [0.08, 0.12, 0.02, 0.08],  # NOGROWTH fails growth
            'debt_to_equity': [0.5, 0.3, 0.2, 1.5]  # DEBT fails debt ratio
        })
        
        filtered = self.scorer.apply_hard_filters(data)
        assert len(filtered) == 0

    def test_calculate_score_components(self):
        """Test individual score component calculations."""
        # Test ROE scoring
        assert self.scorer._score_roe(0.25) == 4  # Excellent
        assert self.scorer._score_roe(0.15) == 3  # Good
        assert self.scorer._score_roe(0.10) == 2  # Average
        assert self.scorer._score_roe(0.05) == 1  # Below average
        assert self.scorer._score_roe(0.02) == 0  # Poor
        
        # Test revenue CAGR scoring
        assert self.scorer._score_revenue_cagr(0.25) == 4  # Excellent
        assert self.scorer._score_revenue_cagr(0.15) == 3  # Good
        assert self.scorer._score_revenue_cagr(0.10) == 2  # Average
        assert self.scorer._score_revenue_cagr(0.05) == 1  # Below average
        assert self.scorer._score_revenue_cagr(0.02) == 0  # Poor
        
        # Test FCF margin scoring
        assert self.scorer._score_fcf_margin(0.25) == 4  # Excellent
        assert self.scorer._score_fcf_margin(0.15) == 3  # Good
        assert self.scorer._score_fcf_margin(0.10) == 2  # Average
        assert self.scorer._score_fcf_margin(0.05) == 1  # Below average
        assert self.scorer._score_fcf_margin(0.02) == 0  # Poor
        
        # Test PEG ratio scoring (lower is better)
        assert self.scorer._score_peg_ratio(0.5) == 4  # Excellent
        assert self.scorer._score_peg_ratio(1.0) == 3  # Good
        assert self.scorer._score_peg_ratio(1.5) == 2  # Average
        assert self.scorer._score_peg_ratio(2.0) == 1  # Below average
        assert self.scorer._score_peg_ratio(3.0) == 0  # Poor
        
        # Test gross margin stability scoring (lower std dev is better)
        assert self.scorer._score_gross_margin_stability(0.01) == 4  # Very stable
        assert self.scorer._score_gross_margin_stability(0.03) == 3  # Stable
        assert self.scorer._score_gross_margin_stability(0.05) == 2  # Moderate
        assert self.scorer._score_gross_margin_stability(0.08) == 1  # Unstable
        assert self.scorer._score_gross_margin_stability(0.12) == 0  # Very unstable

    def test_calculate_scores(self):
        """Test composite score calculation."""
        data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'roe_5yr': [0.25, 0.20],
            'revenue_cagr_5yr': [0.15, 0.12],
            'fcf_margin': [0.20, 0.25],
            'peg_ratio': [1.0, 0.8],
            'gross_margin_std_5yr': [0.02, 0.01]
        })
        
        scored = self.scorer.calculate_scores(data)
        
        # Check that scores are calculated
        assert 'composite_score' in scored.columns
        assert all(scored['composite_score'] >= 0)
        assert all(scored['composite_score'] <= 4.0)
        
        # MSFT should have higher score due to better PEG and stability
        msft_score = scored[scored['symbol'] == 'MSFT']['composite_score'].iloc[0]
        aapl_score = scored[scored['symbol'] == 'AAPL']['composite_score'].iloc[0]
        assert msft_score > aapl_score

    def test_select_top_stocks(self):
        """Test top stock selection."""
        data = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D', 'E'],
            'composite_score': [3.5, 2.8, 3.2, 2.1, 3.8]
        })
        
        top_3 = self.scorer.select_top_stocks(data, n=3)
        
        assert len(top_3) == 3
        # Should be sorted by score descending
        expected_order = ['E', 'A', 'C']  # Scores: 3.8, 3.5, 3.2
        assert list(top_3['symbol']) == expected_order

    def test_check_sell_conditions_broken_thesis(self):
        """Test broken thesis sell condition."""
        # Mock universe data for percentile calculation
        universe_data = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(100)],
            'composite_score': np.linspace(1.0, 4.0, 100)
        })
        
        # Stock with negative revenue growth and low score (bottom 30%)
        stock_data = {
            'symbol': 'TEST',
            'revenue_growth_yoy': -0.05,
            'composite_score': 1.5  # This should be in bottom 30%
        }
        
        result = self.scorer.check_sell_conditions(
            stock_data, universe_data, {}, 0.10
        )
        
        assert 'broken_thesis' in result
        assert result['broken_thesis'] is True

    def test_check_sell_conditions_overvaluation(self):
        """Test overvaluation sell condition."""
        # Mock industry data
        industry_data = pd.DataFrame({
            'symbol': [f'PEER_{i}' for i in range(20)],
            'forward_pe': np.linspace(10, 30, 20)
        })
        
        # Stock with high PE and PEG
        stock_data = {
            'symbol': 'TEST',
            'forward_pe': 35,  # Above 90th percentile of industry
            'peg_ratio': 3.0   # Above 2.5 threshold
        }
        
        result = self.scorer.check_sell_conditions(
            stock_data, pd.DataFrame(), industry_data, 0.10
        )
        
        assert 'overvaluation' in result
        assert result['overvaluation'] is True

    def test_check_sell_conditions_concentration(self):
        """Test concentration sell condition."""
        stock_data = {
            'symbol': 'TEST',
            'revenue_growth_yoy': 0.08,
            'composite_score': 3.0,
            'forward_pe': 15,
            'peg_ratio': 1.2
        }
        
        # Portfolio weight above concentration threshold
        portfolio_weight = 0.25  # Above 20% threshold
        
        result = self.scorer.check_sell_conditions(
            stock_data, pd.DataFrame(), pd.DataFrame(), portfolio_weight
        )
        
        assert 'concentration' in result
        assert result['concentration'] is True

    def test_check_sell_conditions_no_sell(self):
        """Test case where no sell conditions are triggered."""
        universe_data = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(100)],
            'composite_score': np.linspace(1.0, 4.0, 100)
        })
        
        industry_data = pd.DataFrame({
            'symbol': [f'PEER_{i}' for i in range(20)],
            'forward_pe': np.linspace(10, 30, 20)
        })
        
        # Healthy stock
        stock_data = {
            'symbol': 'TEST',
            'revenue_growth_yoy': 0.08,
            'composite_score': 3.5,  # High score
            'forward_pe': 18,        # Reasonable PE
            'peg_ratio': 1.2         # Reasonable PEG
        }
        
        result = self.scorer.check_sell_conditions(
            stock_data, universe_data, industry_data, 0.10  # Low weight
        )
        
        assert result['broken_thesis'] is False
        assert result['overvaluation'] is False
        assert result['concentration'] is False

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty dataframe
        empty_df = pd.DataFrame()
        filtered = self.scorer.apply_hard_filters(empty_df)
        assert len(filtered) == 0
        
        # Missing columns
        incomplete_data = pd.DataFrame({
            'symbol': ['TEST'],
            'market_cap': [3e12]
            # Missing other required columns
        })
        
        with pytest.raises(KeyError):
            self.scorer.apply_hard_filters(incomplete_data)
        
        # NaN values
        nan_data = pd.DataFrame({
            'symbol': ['TEST'],
            'market_cap': [np.nan],
            'net_income_ttm': [1e11],
            'revenue_growth_yoy': [0.08],
            'debt_to_equity': [0.5]
        })
        
        filtered = self.scorer.apply_hard_filters(nan_data)
        assert len(filtered) == 0  # Should filter out NaN values

    def test_score_normalization(self):
        """Test that scores are properly normalized to 0-4 range."""
        # Extreme values
        extreme_data = pd.DataFrame({
            'symbol': ['EXTREME'],
            'roe_5yr': [2.0],  # Very high ROE
            'revenue_cagr_5yr': [-0.5],  # Negative growth
            'fcf_margin': [0.8],  # Very high margin
            'peg_ratio': [10.0],  # Very high PEG
            'gross_margin_std_5yr': [0.5]  # Very unstable
        })
        
        scored = self.scorer.calculate_scores(extreme_data)
        score = scored['composite_score'].iloc[0]
        
        assert 0 <= score <= 4.0

    def test_weight_application(self):
        """Test that weights are correctly applied in composite score."""
        # Create scorer with known weights
        test_config = self.config.copy()
        test_config['scoring']['weights'] = {
            'roe_5yr': 1.0,  # Only ROE matters
            'revenue_cagr_5yr': 0.0,
            'fcf_margin': 0.0,
            'peg_ratio': 0.0,
            'gross_margin_stability': 0.0
        }
        
        test_scorer = FundamentalScorer(test_config)
        
        data = pd.DataFrame({
            'symbol': ['TEST'],
            'roe_5yr': [0.25],  # Should get score of 4
            'revenue_cagr_5yr': [0.02],  # Should get score of 0
            'fcf_margin': [0.02],  # Should get score of 0
            'peg_ratio': [10.0],  # Should get score of 0
            'gross_margin_std_5yr': [0.5]  # Should get score of 0
        })
        
        scored = test_scorer.calculate_scores(data)
        
        # Since only ROE has weight 1.0 and gets score 4, composite should be 4
        assert abs(scored['composite_score'].iloc[0] - 4.0) < 0.01