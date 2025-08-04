#!/usr/bin/env python3
"""
Simple script to test the LLM Query Retrieval System API
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "test_bearer_token_12345"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_main_api():
    """Test the main API endpoint with authentication"""
    print("\n🔍 Testing main API endpoint...")
    
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Simple test payload
    payload = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": [
            "What is this document about?",
            "What is the main content?"
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ API call successful!")
            print(f"📄 Response: {response.json()}")
        elif response.status_code == 401:
            print("❌ Authentication failed!")
            print(f"📄 Error: {response.text}")
        else:
            print(f"⚠️ Unexpected status code: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (this is normal for document processing)")
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing LLM Query Retrieval System API")
    print("=" * 50)
    
    # Test health endpoint first
    if test_health_endpoint():
        # Test main API
        test_main_api()
    else:
        print("❌ Server is not responding. Make sure it's running with: python main.py")