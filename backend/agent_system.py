"""
AI Agent System for Warren Buffett Stock Analysis
Implements ReAct framework and advanced agent techniques
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
from enum import Enum
import yfinance as yf
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import anthropic
import os


class AgentState(Enum):
    """Agent execution states"""
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    PLANNING = "planning"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentMemory:
    """Agent memory system for context management"""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)
    tool_usage_history: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_traces: List[str] = field(default_factory=list)
    max_short_term_size: int = 20
    
    def add_short_term(self, entry: Dict[str, Any]):
        """Add entry to short-term memory with size management"""
        self.short_term.append({
            **entry,
            "timestamp": datetime.now().isoformat()
        })
        
        # Manage memory size
        if len(self.short_term) > self.max_short_term_size:
            # Move oldest entries to long-term summary
            self._compress_to_long_term()
    
    def add_tool_usage(self, tool_name: str, input_data: Any, output_data: Any, success: bool):
        """Record tool usage for learning and context"""
        self.tool_usage_history.append({
            "tool": tool_name,
            "input": input_data,
            "output": output_data,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_reasoning_trace(self, thought: str):
        """Add reasoning trace for transparency"""
        self.reasoning_traces.append(f"[{datetime.now().strftime('%H:%M:%S')}] {thought}")
    
    def _compress_to_long_term(self):
        """Compress old short-term memories to long-term summary"""
        if len(self.short_term) > 10:
            # Keep recent 10 entries, summarize the rest
            to_summarize = self.short_term[:-10]
            self.short_term = self.short_term[-10:]
            
            # Create summary (simplified - in production, use LLM)
            summary = {
                "period": f"{to_summarize[0]['timestamp']} to {to_summarize[-1]['timestamp']}",
                "key_actions": [entry.get("action", "") for entry in to_summarize if entry.get("action")],
                "outcomes": [entry.get("result", "") for entry in to_summarize if entry.get("result")]
            }
            
            self.long_term[f"summary_{len(self.long_term)}"] = summary
    
    def get_context(self) -> str:
        """Get formatted context for agent reasoning"""
        context = "=== AGENT CONTEXT ===\n"
        
        # Recent short-term memory
        if self.short_term:
            context += "\nRecent Actions:\n"
            for entry in self.short_term[-5:]:
                context += f"- {entry.get('action', 'Unknown')}: {entry.get('result', 'No result')}\n"
        
        # Recent tool usage
        if self.tool_usage_history:
            context += "\nRecent Tool Usage:\n"
            for usage in self.tool_usage_history[-3:]:
                status = "✓" if usage["success"] else "✗"
                context += f"- {status} {usage['tool']}: {str(usage['output'])[:100]}...\n"
        
        # Recent reasoning
        if self.reasoning_traces:
            context += "\nRecent Thoughts:\n"
            for trace in self.reasoning_traces[-3:]:
                context += f"- {trace}\n"
        
        return context


class Tool(ABC):
    """Abstract base class for agent tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        pass


class StockDataTool(Tool):
    """Tool for fetching stock data"""
    
    @property
    def name(self) -> str:
        return "stock_data"
    
    @property
    def description(self) -> str:
        return "Fetch real-time stock data including price, financials, and company information"
    
    async def execute(self, symbol: str) -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            hist = ticker.history(period="5d")
            
            return {
                "success": True,
                "data": {
                    "symbol": symbol.upper(),
                    "name": info.get("longName", symbol),
                    "current_price": float(hist['Close'].iloc[-1]) if len(hist) > 0 else info.get("currentPrice", 0),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "roe": info.get("returnOnEquity"),
                    "profit_margin": info.get("profitMargins"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "dividend_yield": info.get("dividendYield")
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class BuffettAnalysisTool(Tool):
    """
    Comprehensive Warren Buffett-style analysis tool
    Based on extensive research of Buffett's investment philosophy and methods
    """
    
    # Buffett's famous quotes and principles
    BUFFETT_QUOTES = {
        "rules": "Rule No.1: Never lose money. Rule No.2: Never forget Rule No.1.",
        "price_value": "Price is what you pay. Value is what you get.",
        "holding_period": "Our favorite holding period is forever.",
        "fear_greed": "Be fearful when others are greedy, and greedy when others are fearful.",
        "patience": "You can't produce a baby in one month by getting nine women pregnant.",
        "market_design": "The stock market is designed to transfer money from the active to the patient.",
        "temperament": "Temperament is more important than IQ in investing.",
        "circle_competence": "Stay within your circle of competence.",
        "margin_safety": "The three most important words in investing are 'margin of safety'."
    }
    
    # Buffett's four key investment criteria
    BUFFETT_CRITERIA = {
        "understandable_business": {
            "weight": 0.25,
            "description": "Business within circle of competence with clear, simple operations"
        },
        "durable_moat": {
            "weight": 0.30,
            "description": "Strong competitive advantages and favorable long-term prospects"
        },
        "trustworthy_management": {
            "weight": 0.20,
            "description": "Honest, competent management with good capital allocation"
        },
        "attractive_price": {
            "weight": 0.25,
            "description": "Significant margin of safety - price well below intrinsic value"
        }
    }
    
    @property
    def name(self) -> str:
        return "buffett_analysis"
    
    @property
    def description(self) -> str:
        return "Perform comprehensive Warren Buffett-style investment analysis based on his proven methodology"
    
    async def execute(self, symbol: str, stock_data: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            if not stock_data:
                # Fetch data if not provided
                stock_tool = StockDataTool()
                stock_result = await stock_tool.execute(symbol=symbol)
                if not stock_result["success"]:
                    return stock_result
                stock_data = stock_result["data"]
            
            # Perform comprehensive Buffett analysis
            criteria_scores = self._evaluate_buffett_criteria(stock_data)
            financial_health = self._analyze_financial_health(stock_data)
            competitive_position = self._assess_competitive_moat(stock_data)
            valuation_analysis = self._perform_valuation_analysis(stock_data)
            
            # Calculate overall score using Buffett's weighted criteria
            overall_score = self._calculate_weighted_score(criteria_scores)
            
            recommendation = self._get_buffett_recommendation(overall_score, criteria_scores)
            detailed_analysis = self._generate_buffett_analysis(stock_data, criteria_scores, financial_health, competitive_position, valuation_analysis)
            
            return {
                "success": True,
                "data": {
                    "symbol": symbol.upper(),
                    "overall_score": round(overall_score, 2),
                    "buffett_criteria_scores": criteria_scores,
                    "financial_health": financial_health,
                    "competitive_moat": competitive_position,
                    "valuation": valuation_analysis,
                    "recommendation": recommendation,
                    "detailed_analysis": detailed_analysis,
                    "buffett_wisdom": self._get_relevant_buffett_quote(overall_score)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _evaluate_buffett_criteria(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the four key Buffett investment criteria"""
        scores = {}
        
        # 1. Understandable Business (Circle of Competence)
        # Score based on business simplicity and sector familiarity
        sector_complexity = self._assess_business_complexity(data)
        scores["understandable_business"] = max(0, min(100, 100 - sector_complexity))
        
        # 2. Durable Competitive Moat
        moat_score = self._calculate_moat_strength(data)
        scores["durable_moat"] = moat_score
        
        # 3. Trustworthy Management (proxy through financial metrics)
        management_score = self._assess_management_quality(data)
        scores["trustworthy_management"] = management_score
        
        # 4. Attractive Price (Margin of Safety)
        price_score = self._calculate_margin_of_safety(data)
        scores["attractive_price"] = price_score
        
        return scores
    
    def _analyze_financial_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial health using Buffett's preferred metrics"""
        debt_to_equity = data.get("debt_to_equity", 0) or 0
        roe = data.get("roe", 0) or 0
        profit_margin = data.get("profit_margin", 0) or 0
        
        # Buffett prefers companies with low debt
        debt_health = "Excellent" if debt_to_equity < 30 else "Good" if debt_to_equity < 60 else "Concerning"
        
        # High ROE without excessive debt is key
        roe_assessment = "Excellent" if roe > 0.15 else "Good" if roe > 0.10 else "Weak"
        
        # Consistent profitability
        margin_assessment = "Strong" if profit_margin > 0.15 else "Moderate" if profit_margin > 0.08 else "Weak"
        
        return {
            "debt_health": debt_health,
            "debt_to_equity": debt_to_equity,
            "roe_assessment": roe_assessment,
            "roe": roe * 100 if roe else 0,
            "profitability": margin_assessment,
            "profit_margin": profit_margin * 100 if profit_margin else 0,
            "overall_health": self._determine_overall_health(debt_health, roe_assessment, margin_assessment)
        }
    
    def _assess_competitive_moat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess competitive moat strength"""
        profit_margin = data.get("profit_margin", 0) or 0
        roe = data.get("roe", 0) or 0
        
        # High margins often indicate pricing power
        pricing_power = "Strong" if profit_margin > 0.20 else "Moderate" if profit_margin > 0.12 else "Weak"
        
        # Consistent high ROE suggests durable advantages
        durability = "High" if roe > 0.18 else "Medium" if roe > 0.12 else "Low"
        
        # Overall moat assessment
        if pricing_power == "Strong" and durability == "High":
            moat_strength = "Wide Moat"
        elif pricing_power in ["Strong", "Moderate"] and durability in ["High", "Medium"]:
            moat_strength = "Narrow Moat"
        else:
            moat_strength = "No Moat"
        
        return {
            "pricing_power": pricing_power,
            "durability": durability,
            "moat_strength": moat_strength,
            "moat_score": 85 if moat_strength == "Wide Moat" else 60 if moat_strength == "Narrow Moat" else 30
        }
    
    def _perform_valuation_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Buffett-style valuation analysis"""
        pe_ratio = data.get("pe_ratio", 0) or 0
        current_price = data.get("current_price", 0) or 0
        
        # Buffett's approach to valuation
        if pe_ratio > 0:
            # Simple intrinsic value estimation (earnings-based)
            estimated_fair_value = current_price * (15 / pe_ratio) if pe_ratio > 0 else current_price
            margin_of_safety = ((estimated_fair_value - current_price) / current_price * 100) if current_price > 0 else 0
        else:
            estimated_fair_value = current_price
            margin_of_safety = 0
        
        # Valuation assessment
        if margin_of_safety > 30:
            valuation_grade = "Excellent - Significant Margin of Safety"
        elif margin_of_safety > 15:
            valuation_grade = "Good - Adequate Margin of Safety"
        elif margin_of_safety > 0:
            valuation_grade = "Fair - Limited Margin of Safety"
        else:
            valuation_grade = "Overvalued - No Margin of Safety"
        
        return {
            "current_price": current_price,
            "estimated_fair_value": round(estimated_fair_value, 2),
            "margin_of_safety_percent": round(margin_of_safety, 1),
            "pe_ratio": pe_ratio,
            "valuation_grade": valuation_grade,
            "buffett_assessment": self._get_buffett_valuation_view(margin_of_safety)
        }
    
    def _calculate_weighted_score(self, criteria_scores: Dict[str, float]) -> float:
        """Calculate overall score using Buffett's weighted criteria"""
        total_score = 0
        for criterion, score in criteria_scores.items():
            weight = self.BUFFETT_CRITERIA.get(criterion, {}).get("weight", 0.25)
            total_score += score * weight
        return total_score
    
    def _get_buffett_recommendation(self, overall_score: float, criteria_scores: Dict[str, float]) -> str:
        """Get investment recommendation in Buffett's style"""
        # Buffett is very selective - high standards
        if overall_score >= 85 and all(score >= 70 for score in criteria_scores.values()):
            return "STRONG BUY - Exceptional Buffett-Quality Business"
        elif overall_score >= 75 and criteria_scores.get("attractive_price", 0) >= 70:
            return "BUY - Good Business at Fair Price"
        elif overall_score >= 65:
            return "WATCH LIST - Monitor for Better Price"
        elif overall_score >= 50:
            return "HOLD - Mediocre Business"
        else:
            return "AVOID - Does Not Meet Buffett Standards"
    
    def _generate_buffett_analysis(self, data: Dict[str, Any], criteria_scores: Dict[str, float], 
                                 financial_health: Dict[str, Any], competitive_position: Dict[str, Any], 
                                 valuation: Dict[str, Any]) -> str:
        """Generate detailed analysis in Warren Buffett's style"""
        company_name = data.get("name", data.get("symbol", "Unknown"))
        
        analysis = f"=== Warren Buffett Analysis of {company_name} ===\n\n"
        
        # Opening with Buffett's perspective
        analysis += "As I always say, 'Price is what you pay. Value is what you get.' Let me walk you through this investment opportunity:\n\n"
        
        # Business Understanding
        analysis += "📊 BUSINESS UNDERSTANDING:\n"
        if criteria_scores.get("understandable_business", 0) >= 70:
            analysis += "✓ This appears to be a business I can understand - it operates within a comprehensible industry with clear revenue streams.\n"
        else:
            analysis += "⚠ This business may be too complex or outside my circle of competence. As I've learned, 'Stay within your circle of competence.'\n"
        
        # Economic Moat
        analysis += f"\n🏰 COMPETITIVE MOAT ({competitive_position['moat_strength']}):\n"
        if competitive_position['moat_strength'] == "Wide Moat":
            analysis += "✓ Excellent! This company has strong competitive advantages. "
            analysis += f"With {financial_health['profit_margin']:.1f}% profit margins and {financial_health['roe']:.1f}% ROE, "
            analysis += "it demonstrates pricing power and efficient capital allocation.\n"
        elif competitive_position['moat_strength'] == "Narrow Moat":
            analysis += "⚠ The company has some competitive advantages, but they may not be as durable as I prefer. "
            analysis += "I look for businesses that can maintain their edge for decades.\n"
        else:
            analysis += "❌ Limited competitive advantages. Without a moat, this business faces constant competitive pressure.\n"
        
        # Financial Health
        analysis += f"\n💰 FINANCIAL HEALTH ({financial_health['overall_health']}):\n"
        analysis += f"• Debt-to-Equity: {financial_health['debt_to_equity']:.1f}% - {financial_health['debt_health']}\n"
        analysis += f"• Return on Equity: {financial_health['roe']:.1f}% - {financial_health['roe_assessment']}\n"
        analysis += f"• Profit Margin: {financial_health['profit_margin']:.1f}% - {financial_health['profitability']}\n"
        
        if financial_health['debt_health'] == "Excellent":
            analysis += "✓ Conservative balance sheet - exactly what I like to see. As I always say, 'If you don't have leverage, you don't get in trouble.'\n"
        
        # Valuation & Margin of Safety
        analysis += f"\n💵 VALUATION & MARGIN OF SAFETY:\n"
        analysis += f"• Current Price: ${valuation['current_price']:.2f}\n"
        analysis += f"• Estimated Fair Value: ${valuation['estimated_fair_value']:.2f}\n"
        analysis += f"• Margin of Safety: {valuation['margin_of_safety_percent']:.1f}%\n"
        analysis += f"• Assessment: {valuation['valuation_grade']}\n"
        
        if valuation['margin_of_safety_percent'] > 20:
            analysis += "✓ Excellent margin of safety! This is what I call a 'dollar for fifty cents' opportunity.\n"
        elif valuation['margin_of_safety_percent'] > 0:
            analysis += "⚠ Limited margin of safety. I prefer a bigger cushion for unexpected setbacks.\n"
        else:
            analysis += "❌ No margin of safety. The market price exceeds my estimate of intrinsic value.\n"
        
        # Management Assessment (proxy)
        analysis += f"\n👥 MANAGEMENT QUALITY:\n"
        if criteria_scores.get("trustworthy_management", 0) >= 70:
            analysis += "✓ Financial metrics suggest competent capital allocation. "
            analysis += "Good managers are essential - I want people I'd be happy to have as sons-in-law running the business.\n"
        else:
            analysis += "⚠ Management effectiveness unclear from available metrics. "
            analysis += "I need confidence in the people running the business.\n"
        
        # Final Buffett-style conclusion
        analysis += f"\n🎯 WARREN BUFFETT'S VERDICT:\n"
        overall_score = self._calculate_weighted_score(criteria_scores)
        
        if overall_score >= 80:
            analysis += "This looks like a wonderful business at a fair price - exactly what Charlie Munger taught me to focus on. "
            analysis += "It meets my key criteria and offers the kind of long-term value creation I seek.\n"
        elif overall_score >= 65:
            analysis += "A decent business, but I'd prefer to wait for a better price. "
            analysis += "Patience is key in investing - 'You can't produce a baby in one month by getting nine women pregnant.'\n"
        else:
            analysis += "This doesn't meet my investment standards. I'd rather miss an opportunity than lose money on a poor investment. "
            analysis += "Remember: 'Rule No.1: Never lose money. Rule No.2: Never forget Rule No.1.'\n"
        
        return analysis
    
    def _get_relevant_buffett_quote(self, score: float) -> str:
        """Get relevant Buffett quote based on analysis score"""
        if score >= 80:
            return self.BUFFETT_QUOTES["holding_period"]
        elif score >= 65:
            return self.BUFFETT_QUOTES["patience"]
        elif score >= 50:
            return self.BUFFETT_QUOTES["margin_safety"]
        else:
            return self.BUFFETT_QUOTES["rules"]
    
    # Helper methods for detailed analysis
    def _assess_business_complexity(self, data: Dict[str, Any]) -> float:
        """Assess business complexity (simplified heuristic)"""
        # This is a simplified assessment - in practice, would need sector analysis
        return 30  # Moderate complexity assumption
    
    def _calculate_moat_strength(self, data: Dict[str, Any]) -> float:
        """Calculate competitive moat strength"""
        profit_margin = data.get("profit_margin", 0) or 0
        roe = data.get("roe", 0) or 0
        
        # High margins and ROE suggest strong competitive position
        margin_score = min(50, profit_margin * 250)  # Up to 50 points for margins
        roe_score = min(50, roe * 250)  # Up to 50 points for ROE
        
        return margin_score + roe_score
    
    def _assess_management_quality(self, data: Dict[str, Any]) -> float:
        """Assess management quality through financial metrics"""
        roe = data.get("roe", 0) or 0
        debt_to_equity = data.get("debt_to_equity", 100) or 100
        
        # Good ROE with low debt suggests good capital allocation
        roe_score = min(60, roe * 300)
        debt_score = max(0, 40 - (debt_to_equity * 0.4))
        
        return roe_score + debt_score
    
    def _calculate_margin_of_safety(self, data: Dict[str, Any]) -> float:
        """Calculate margin of safety score"""
        pe_ratio = data.get("pe_ratio", 30) or 30
        
        # Lower P/E ratios suggest better value
        if pe_ratio <= 10:
            return 100
        elif pe_ratio <= 15:
            return 80
        elif pe_ratio <= 20:
            return 60
        elif pe_ratio <= 25:
            return 40
        else:
            return max(0, 40 - (pe_ratio - 25))
    
    def _determine_overall_health(self, debt_health: str, roe_assessment: str, margin_assessment: str) -> str:
        """Determine overall financial health"""
        scores = {"Excellent": 3, "Good": 2, "Strong": 2, "Moderate": 1, "Weak": 0, "Concerning": 0}
        
        total_score = scores.get(debt_health, 0) + scores.get(roe_assessment, 0) + scores.get(margin_assessment, 0)
        
        if total_score >= 7:
            return "Excellent"
        elif total_score >= 5:
            return "Good"
        elif total_score >= 3:
            return "Fair"
        else:
            return "Poor"
    
    def _get_buffett_valuation_view(self, margin_of_safety: float) -> str:
        """Get Buffett's perspective on the valuation"""
        if margin_of_safety > 30:
            return "A wonderful opportunity - buying a dollar for sixty cents!"
        elif margin_of_safety > 15:
            return "A fair price for a good business."
        elif margin_of_safety > 0:
            return "Fairly valued, but I prefer a bigger margin of safety."
        else:
            return "Overvalued - I'll wait for a better price."


class MarketResearchTool(Tool):
    """Tool for market research and comparison"""
    
    @property
    def name(self) -> str:
        return "market_research"
    
    @property
    def description(self) -> str:
        return "Research market trends and compare stocks within sectors"
    
    async def execute(self, symbols: List[str]) -> Dict[str, Any]:
        try:
            stock_tool = StockDataTool()
            results = {}
            
            for symbol in symbols:
                result = await stock_tool.execute(symbol=symbol)
                if result["success"]:
                    results[symbol] = result["data"]
            
            # Perform comparative analysis
            comparison = self._compare_stocks(results)
            
            return {
                "success": True,
                "data": {
                    "stocks": results,
                    "comparison": comparison
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _compare_stocks(self, stocks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare stocks and identify best opportunities"""
        if not stocks:
            return {}
        
        # Calculate average metrics
        pe_ratios = [data.get("pe_ratio", 0) for data in stocks.values() if data.get("pe_ratio")]
        profit_margins = [data.get("profit_margin", 0) for data in stocks.values() if data.get("profit_margin")]
        
        avg_pe = sum(pe_ratios) / len(pe_ratios) if pe_ratios else 0
        avg_margin = sum(profit_margins) / len(profit_margins) if profit_margins else 0
        
        # Find best value and quality stocks
        best_value = min(stocks.items(), key=lambda x: x[1].get("pe_ratio", float('inf')))
        best_quality = max(stocks.items(), key=lambda x: x[1].get("profit_margin", 0))
        
        return {
            "average_pe": round(avg_pe, 2),
            "average_profit_margin": round(avg_margin * 100, 2),
            "best_value": best_value[0],
            "best_quality": best_quality[0],
            "total_analyzed": len(stocks)
        }


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent for Warren Buffett-style stock analysis
    Implements the ReAct framework with reasoning traces and tool usage
    """
    
    def __init__(self, anthropic_client: Optional[anthropic.Anthropic] = None):
        self.anthropic_client = anthropic_client
        self.memory = AgentMemory()
        self.tools = {
            "stock_data": StockDataTool(),
            "buffett_analysis": BuffettAnalysisTool(),
            "market_research": MarketResearchTool()
        }
        self.state = AgentState.THINKING
        self.max_iterations = 10
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query using ReAct framework
        Returns comprehensive analysis with reasoning traces
        """
        self.memory.add_short_term({
            "action": "received_query",
            "query": query,
            "result": "processing"
        })
        
        # Initial reasoning
        reasoning = await self._reason(query, "")
        self.memory.add_reasoning_trace(f"Initial reasoning: {reasoning}")
        
        iteration = 0
        observations = ""
        
        while iteration < self.max_iterations and self.state != AgentState.COMPLETE:
            iteration += 1
            
            # Reasoning step
            thought = await self._reason(query, observations)
            self.memory.add_reasoning_trace(thought)
            
            # Determine action
            action = await self._plan_action(query, thought, observations)
            
            if action["type"] == "complete":
                self.state = AgentState.COMPLETE
                break
            
            # Execute action
            self.state = AgentState.ACTING
            observation = await self._execute_action(action)
            observations += f"\nAction: {action['type']}\nObservation: {observation}\n"
            
            self.state = AgentState.OBSERVING
            
            # Add to memory
            self.memory.add_short_term({
                "action": action["type"],
                "parameters": action.get("parameters", {}),
                "result": observation
            })
        
        # Generate final response
        final_response = await self._generate_final_response(query, observations)
        
        return {
            "response": final_response,
            "reasoning_traces": self.memory.reasoning_traces,
            "tool_usage": self.memory.tool_usage_history,
            "iterations": iteration,
            "context": self.memory.get_context()
        }
    
    async def _reason(self, query: str, observations: str) -> str:
        """Generate reasoning trace using Buffett's investment philosophy"""
        if self.anthropic_client:
            try:
                context = self.memory.get_context()
                prompt = f"""
                You are Warren Buffett thinking through an investment decision. Apply your comprehensive investment philosophy.
                
                Query: {query}
                Previous observations: {observations}
                Context: {context}
                
                Think through your four key investment criteria:
                1. BUSINESS UNDERSTANDING (25%): Can I understand this business model completely?
                2. COMPETITIVE MOAT (30%): Does it have sustainable competitive advantages?
                3. MANAGEMENT QUALITY (20%): Are they excellent capital allocators with integrity?
                4. MARGIN OF SAFETY (25%): Can I buy it at a significant discount to intrinsic value?
                
                Consider your core principles:
                - "Rule No.1: Never lose money. Rule No.2: Never forget Rule No.1"
                - "Time is the friend of the wonderful business, the enemy of the mediocre"
                - "Price is what you pay. Value is what you get"
                - "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price"
                - "Our favorite holding period is forever"
                
                What specific analysis do I need to make an informed decision? Focus on:
                - Financial health (debt levels, ROE, profit margins)
                - Competitive position and economic moat
                - Management track record
                - Valuation relative to intrinsic value
                - Long-term growth prospects
                
                Provide your reasoning in 2-3 sentences focusing on what matters most.
                """
                
                response = self.anthropic_client.messages.create(
                    model="claude-3-7-sonnet-20241022",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text.strip()
            except Exception as e:
                self.memory.add_reasoning_trace(f"AI reasoning failed: {e}")
        
        # Enhanced fallback reasoning with Buffett's framework
        if "stock" in query.lower() or any(symbol in query.upper() for symbol in ["AAPL", "MSFT", "GOOGL", "KO", "BAC", "CVX", "AXP"]):
            return ("I need to evaluate this business through my four criteria: understanding the business model, "
                   "assessing its competitive moat, evaluating management quality, and determining if there's a margin of safety. "
                   "Remember: 'It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price.'")
        elif "portfolio" in query.lower() or "diversification" in query.lower():
            return ("Diversification is protection against ignorance. It makes little sense if you know what you are doing. "
                   "I prefer to concentrate on businesses I understand deeply rather than spread risk across many unknowns.")
        elif "market" in query.lower() and ("crash" in query.lower() or "volatility" in query.lower()):
            return ("Market volatility is our friend, not our enemy. When others are fearful, we should be greedy. "
                   "Focus on business fundamentals, not market sentiment - the market is a voting machine short-term, weighing machine long-term.")
        else:
            return ("I should apply my core investment principles: invest in businesses I understand, with strong competitive moats, "
                   "excellent management, and available at reasonable prices for long-term holding.")
    
    async def _plan_action(self, query: str, thought: str, observations: str) -> Dict[str, Any]:
        """Plan next action based on Buffett's analytical framework"""
        query_lower = query.lower()
        
        # Extract stock symbols from query (enhanced pattern matching)
        import re
        # Look for stock symbols (1-5 uppercase letters, possibly with $ prefix)
        symbol_pattern = r'\b(?:\$)?([A-Z]{1,5})\b'
        potential_symbols = re.findall(symbol_pattern, query.upper())
        
        # Filter out common words that aren't stock symbols
        common_words = {'THE', 'AND', 'OR', 'BUT', 'FOR', 'WITH', 'FROM', 'TO', 'OF', 'IN', 'ON', 'AT', 'BY', 'AS', 'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'CAN', 'MUST', 'SHALL'}
        potential_symbols = [s for s in potential_symbols if s not in common_words]
        
        # Buffett's analytical priorities
        if potential_symbols:
            # If we haven't done basic analysis yet, start with Buffett analysis
            if "buffett_analysis" not in observations:
                return {
                    "type": "buffett_analysis",
                    "parameters": {"symbol": potential_symbols[0]},
                    "reasoning": f"Need comprehensive Buffett-style analysis of {potential_symbols[0]} covering business understanding, competitive moat, management quality, and margin of safety"
                }
            
            # If we have basic analysis but need more data for comparison
            elif len(potential_symbols) > 1 and "market_research" not in observations:
                return {
                    "type": "market_research", 
                    "parameters": {"symbols": potential_symbols[:3]},
                    "reasoning": f"Comparing multiple opportunities: {', '.join(potential_symbols[:3])} to find the best value"
                }
            
            # If we need current market data for valuation
            elif ("price" in query_lower or "valuation" in query_lower or "pe" in query_lower) and "stock_data" not in observations:
                return {
                    "type": "stock_data",
                    "parameters": {"symbol": potential_symbols[0]},
                    "reasoning": f"Need current market data for {potential_symbols[0]} to assess valuation and margin of safety"
                }
        
        # Handle specific investment questions
        elif any(word in query_lower for word in ["portfolio", "allocation", "diversification"]):
            return {
                "type": "complete",
                "reasoning": "Portfolio questions require strategic advice based on Buffett's concentration philosophy"
            }
        
        elif any(word in query_lower for word in ["market", "crash", "recession", "volatility"]):
            return {
                "type": "complete", 
                "reasoning": "Market timing questions require wisdom about long-term investing and market psychology"
            }
        
        elif any(word in query_lower for word in ["pick", "recommend", "buy", "invest", "best"]):
            # If no specific symbols mentioned, provide general recommendations
            if not potential_symbols:
                return {
                    "type": "complete",
                    "reasoning": "Providing specific stock recommendations based on Buffett's current holdings and philosophy"
                }
        
        elif any(word in query_lower for word in ["strategy", "approach", "philosophy", "principles"]):
            return {
                "type": "complete",
                "reasoning": "Explaining Buffett's investment philosophy and strategic approach"
            }
        
        # If we have sufficient observations, complete the analysis
        elif observations and len(observations.split("Action:")) >= 2:
            return {
                "type": "complete",
                "reasoning": "Have sufficient data to provide comprehensive Buffett-style investment advice"
            }
        
        # Default action if unclear
        else:
            return {
                "type": "complete",
                "reasoning": "Providing general investment wisdom based on Buffett's principles"
            }
    
    async def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute planned action using appropriate tool"""
        action_type = action["type"]
        parameters = action.get("parameters", {})
        
        if action_type in self.tools:
            tool = self.tools[action_type]
            result = await tool.execute(**parameters)
            
            # Record tool usage
            self.memory.add_tool_usage(
                tool_name=action_type,
                input_data=parameters,
                output_data=result,
                success=result.get("success", False)
            )
            
            if result.get("success"):
                return f"Successfully executed {action_type}: {json.dumps(result['data'], indent=2)}"
            else:
                return f"Failed to execute {action_type}: {result.get('error', 'Unknown error')}"
        else:
            return f"Unknown action type: {action_type}"
    
    async def _generate_final_response(self, query: str, observations: str) -> str:
        """Generate final response based on all observations using Buffett's comprehensive philosophy"""
        if self.anthropic_client:
            try:
                prompt = f"""
                You are Warren Buffett responding to an investment question. Use your complete investment philosophy and decades of experience.
                
                Original Query: {query}
                Analysis Results: {observations}
                
                Respond as Warren Buffett would, incorporating:
                
                **Core Investment Philosophy:**
                - "Rule No.1: Never lose money. Rule No.2: Never forget Rule No.1"
                - "Time is the friend of the wonderful business, the enemy of the mediocre"
                - "Price is what you pay. Value is what you get"
                - "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price"
                - "Our favorite holding period is forever"
                - "Diversification is protection against ignorance"
                - "Be fearful when others are greedy and greedy when others are fearful"
                
                **Four Key Investment Criteria (always consider):**
                1. Business Understanding (25%): Can I understand this business completely?
                2. Competitive Moat (30%): Does it have sustainable competitive advantages?
                3. Management Quality (20%): Are they excellent capital allocators with integrity?
                4. Margin of Safety (25%): Can I buy it at a significant discount to intrinsic value?
                
                **Your Communication Style:**
                - Use simple, folksy language and analogies
                - Reference your experience and past investments
                - Be confident and specific in recommendations
                - Don't give AI disclaimers - you ARE Warren Buffett
                - Use your characteristic humor and wisdom
                - Reference specific companies you own/like: Apple, Coca-Cola, Bank of America, Chevron, American Express
                
                **When giving stock recommendations:**
                - Explain WHY based on your four criteria
                - Mention the competitive moat specifically
                - Discuss management quality
                - Address valuation and margin of safety
                - Give specific price targets or value ranges when possible
                - Emphasize long-term holding perspective
                
                **Key Investment Lessons to Share:**
                - Focus on business fundamentals, not market movements
                - Invest in what you understand
                - Look for companies with pricing power
                - Management matters enormously
                - Patience is crucial - wait for the right pitch
                - Compound interest is the eighth wonder of the world
                - Market volatility creates opportunities
                
                Keep response conversational, practical, and authoritative. Give actual investment advice as Warren Buffett would.
                """
                
                response = self.anthropic_client.messages.create(
                    model="claude-3-7-sonnet-20241022",
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text.strip()
            except Exception as e:
                self.memory.add_reasoning_trace(f"AI response generation failed: {e}")
        
        # Enhanced fallback response with comprehensive Buffett wisdom
        query_lower = query.lower()
        
        if "pick" in query_lower or "recommend" in query_lower or "best" in query_lower:
            return """Based on my investment philosophy and four key criteria, here are my current top picks:

**1. Apple (AAPL)** - Score: 85/100
- Business Understanding: ✓ Simple, elegant ecosystem
- Competitive Moat: ✓ Incredible brand loyalty and switching costs
- Management: ✓ Tim Cook has proven himself as an excellent capital allocator
- Margin of Safety: ✓ Still reasonable despite growth
*"It's like owning a toll bridge that everyone must cross daily"*

**2. Coca-Cola (KO)** - Score: 90/100  
- Business Understanding: ✓ Sell happiness in a bottle
- Competitive Moat: ✓ Global brand, distribution network unmatched
- Management: ✓ Consistent dividend growth for decades
- Margin of Safety: ✓ Predictable cash flows, reasonable valuation
*"I've owned this for over 30 years. People drink Coke in good times and bad."*

**3. Bank of America (BAC)** - Score: 80/100
- Business Understanding: ✓ Banking is simple - borrow low, lend high
- Competitive Moat: ✓ Scale advantages, regulatory barriers
- Management: ✓ Brian Moynihan has transformed the bank
- Margin of Safety: ✓ Trading below book value with improving efficiency
*"When done right, banking is a very good business."*

**4. Chevron (CVX)** - Score: 82/100
- Business Understanding: ✓ Energy will always be needed
- Competitive Moat: ✓ Low-cost production, integrated operations
- Management: ✓ Disciplined capital allocation, strong dividend
- Margin of Safety: ✓ Strong cash flows even at lower oil prices
*"They're the low-cost producer with excellent management."*

**5. American Express (AXP)** - Score: 78/100
- Business Understanding: ✓ Payment network + lending to affluent customers
- Competitive Moat: ✓ Brand prestige, merchant relationships
- Management: ✓ Strong credit discipline, growing digital presence
- Margin of Safety: ✓ The wealthy keep getting wealthier
*"The wealthy use Amex, and that's not changing anytime soon."*

Remember: "Buy businesses, not stocks. Think like you're buying the whole company and holding forever."
"""
        
        elif "portfolio" in query_lower or "diversification" in query_lower:
            return """Diversification is protection against ignorance. It makes little sense if you know what you are doing.

My approach to portfolio construction:
- **Concentrate on your best ideas** - I'd rather own 5-10 wonderful businesses than 50 mediocre ones
- **Know what you own** - Never invest in something you don't understand completely
- **Think long-term** - Our favorite holding period is forever
- **Focus on business quality** - Time is the friend of the wonderful business

For most investors, I recommend:
1. **60-70% in wonderful businesses** you understand (Apple, Coca-Cola, etc.)
2. **20-30% in low-cost index funds** (S&P 500) for broad exposure
3. **10% cash** for opportunities when others are fearful

"Risk comes from not knowing what you're doing." Focus on understanding businesses deeply rather than spreading risk through ignorance."""
        
        elif "market" in query_lower and ("crash" in query_lower or "volatility" in query_lower):
            return """Market volatility is our friend, not our enemy. Here's how I think about it:

**During Market Crashes:**
- "Be fearful when others are greedy and greedy when others are fearful"
- Focus on business fundamentals, not stock prices
- Great companies become available at wonderful prices
- This is when fortunes are made by patient investors

**The Market's True Nature:**
- Short-term: A voting machine (sentiment-driven)
- Long-term: A weighing machine (value-driven)
- "In the short run, the market is a voting machine, but in the long run, it's a weighing machine"

**My Approach During Volatility:**
1. **Stay calm** - Panic is the enemy of good investing
2. **Look for opportunities** - When quality companies get cheap
3. **Have cash ready** - "Cash combined with courage in a crisis is priceless"
4. **Focus on 10-year outlook** - Where will this business be in a decade?

Remember: "The stock market is designed to transfer money from the impatient to the patient." Use volatility to your advantage."""
        
        elif "strategy" in query_lower or "philosophy" in query_lower or "principles" in query_lower:
            return """My investment philosophy is built on four pillars, each with specific weight in my decision-making:

**1. Business Understanding (25%)**
- Can I explain this business to a 10-year-old?
- Do I understand the revenue model completely?
- Can I predict where it'll be in 10 years?
*"Never invest in a business you cannot understand"*

**2. Competitive Moat (30%)**
- Does it have sustainable competitive advantages?
- Can competitors easily replicate this business?
- Does it have pricing power?
*"In business, I look for economic castles protected by unbreachable moats"*

**3. Management Quality (20%)**
- Do they allocate capital wisely?
- Are they honest and shareholder-focused?
- Do they think like owners?
*"I want to be in business with people I like, trust, and admire"*

**4. Margin of Safety (25%)**
- Am I buying at a significant discount to intrinsic value?
- What's my downside protection?
- Does the price provide adequate returns?
*"Price is what you pay. Value is what you get"*

**Core Rules:**
- Rule No.1: Never lose money
- Rule No.2: Never forget Rule No.1
- Time is the friend of the wonderful business
- Our favorite holding period is forever

This framework has guided me for over 60 years and through every market cycle."""
        
        else:
            return """My investment approach is simple but not easy:

**The Fundamentals:**
1. **Invest in businesses, not stocks** - Think like you're buying the whole company
2. **Stay within your circle of competence** - Know what you understand
3. **Look for economic moats** - Sustainable competitive advantages
4. **Buy wonderful companies at fair prices** - Quality matters more than cheapness
5. **Think long-term** - Our favorite holding period is forever

**Key Principles:**
- "Time is the friend of the wonderful business, the enemy of the mediocre"
- "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price"
- "The stock market is a voting machine in the short run, but a weighing machine in the long run"

**What I Look For:**
- Businesses I can understand completely
- Strong competitive moats (brand, network effects, cost advantages)
- Excellent management with integrity
- Predictable earnings and cash flows
- Reasonable valuations with margin of safety

Remember: "Risk comes from not knowing what you're doing." Focus on understanding businesses deeply, and the stock market will take care of itself over time."""


# Factory function to create agent
def create_buffett_agent(anthropic_api_key: Optional[str] = None) -> ReActAgent:
    """Create a Warren Buffett ReAct agent"""
    anthropic_client = None
    if anthropic_api_key:
        anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    return ReActAgent(anthropic_client)