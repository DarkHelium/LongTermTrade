#!/usr/bin/env python3

import requests
import json

def test_backend_api():
    """Test the backend API to verify it's working correctly"""
    
    # Test the chat endpoint
    url = "http://localhost:3000/chat"
    
    # Test data
    test_queries = [
        {
            "query": "What are some good growth stocks under $100B market cap?",
            "is_live": False
        },
        {
            "query": "Tell me about current market trends",
            "is_live": False
        }
    ]
    
    print("Testing Backend API...\n")
    
    for i, test_data in enumerate(test_queries, 1):
        print(f"Test {i}: {test_data['query']}")
        print("-" * 50)
        
        try:
            response = requests.post(url, json=test_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"Response: {result.get('response', 'No response')[:200]}...")
                print(f"Suggestions: {len(result.get('suggestions', []))} items")
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            
        print("\n")

if __name__ == "__main__":
    test_backend_api()