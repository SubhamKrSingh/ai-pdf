#!/usr/bin/env python3
"""
Test script to verify the API works end-to-end.
"""

import requests
import json
import time

def test_api():
    """Test the API with a sample request."""
    
    # API endpoint
    url = "http://localhost:8000/api/v1/hackrx/run"
    
    # Sample request data
    data = {
        "url": "https://example.com/sample.pdf",
        "questions": [
            "What is the main topic of this document?",
            "Can you summarize the key points?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer cbc7c316098602c22ecf59ad563842c04778bcab6dccabf9c6163ffa3dbaaecb"
    }
    
    try:
        print("Making API request...")
        print(f"URL: {url}")
        print(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ API request successful!")
            return True
        else:
            print("❌ API request failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error making API request: {e}")
        return False

if __name__ == "__main__":
    print("Note: Make sure the server is running on localhost:8000")
    print("Run: python main.py")
    print()
    
    # Wait a moment for user to start server
    input("Press Enter when the server is running...")
    
    success = test_api()
    if not success:
        exit(1)