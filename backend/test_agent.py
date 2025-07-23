#!/usr/bin/env python3
"""
Test script for the AI Agent System
"""

import asyncio
import os
from dotenv import load_dotenv
from agent_system import create_buffett_agent

# Load environment variables
load_dotenv()

async def test_agent():
    """Test the ReAct agent system"""
    
    print("🤖 Testing AI Agent System...")
    print("=" * 50)
    
    # Create the agent
    agent = create_buffett_agent()
    
    # Test queries
    test_queries = [
        "What do you think about Apple stock?",
        "Can you analyze Tesla's competitive moat?",
        "What are your top stock picks right now?",
        "How do you evaluate a company's management quality?",
        "Should I invest in cryptocurrency?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            result = await agent.process_query(query)
            
            print(f"✅ Response: {result['response'][:200]}...")
            print(f"🧠 Reasoning Steps: {len(result['reasoning_traces'])}")
            print(f"🔧 Tools Used: {len(result['tool_usage'])}")
            print(f"🔄 Iterations: {result['iterations']}")
            
            if result['reasoning_traces']:
                print("\n📝 Reasoning Traces:")
                for trace in result['reasoning_traces'][:2]:  # Show first 2
                    print(f"  - {trace[:100]}...")
            
            if result['tool_usage']:
                print("\n🛠️ Tool Usage:")
                for tool in result['tool_usage'][:2]:  # Show first 2
                    print(f"  - {tool.get('tool', 'Unknown')}: {tool.get('result', 'No result')[:50]}...")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "=" * 50)

async def test_individual_tools():
    """Test individual tools"""
    
    print("\n🔧 Testing Individual Tools...")
    print("=" * 50)
    
    from agent_system import StockDataTool, BuffettAnalysisTool, MarketResearchTool
    
    # Test StockDataTool
    stock_tool = StockDataTool()
    print("\n📊 Testing StockDataTool with AAPL...")
    try:
        result = await stock_tool.execute("AAPL")
        print(f"✅ Stock data retrieved: {result[:100]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test BuffettAnalysisTool
    analysis_tool = BuffettAnalysisTool()
    print("\n🎯 Testing BuffettAnalysisTool with AAPL...")
    try:
        result = await analysis_tool.execute("AAPL")
        print(f"✅ Analysis completed: {result[:100]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test MarketResearchTool
    research_tool = MarketResearchTool()
    print("\n🔍 Testing MarketResearchTool...")
    try:
        result = await research_tool.execute("technology sector trends")
        print(f"✅ Research completed: {result[:100]}...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Agent System Tests...")
    
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️ Warning: ANTHROPIC_API_KEY not found in environment variables")
        print("The agent will use fallback responses.")
    
    # Run tests
    asyncio.run(test_agent())
    asyncio.run(test_individual_tools())
    
    print("\n✨ Testing completed!")