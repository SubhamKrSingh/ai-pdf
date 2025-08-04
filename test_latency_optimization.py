#!/usr/bin/env python3
"""
Test script to demonstrate latency optimization improvements.
Tests the same PDF URL multiple times to show caching benefits.
"""

import asyncio
import time
import httpx
import json
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "your-auth-token-here"  # Replace with your actual token
TEST_PDF_URL = "https://example.com/test-document.pdf"  # Replace with actual PDF URL

# Test questions
TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key findings mentioned?",
    "What recommendations are provided?",
    "Who are the target audience for this document?",
    "What is the conclusion of this document?"
]


async def make_api_request(session: httpx.AsyncClient, questions: list) -> Dict[str, Any]:
    """Make a single API request and measure response time."""
    start_time = time.time()
    
    try:
        response = await session.post(
            f"{API_BASE_URL}/api/v1/hackrx/run",
            headers={
                "Authorization": f"Bearer {AUTH_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "documents": TEST_PDF_URL,
                "questions": questions
            },
            timeout=120.0  # 2 minute timeout
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "response_time": response_time,
                "answers_count": len(data.get("answers", [])),
                "status_code": response.status_code
            }
        else:
            return {
                "success": False,
                "response_time": response_time,
                "error": response.text,
                "status_code": response.status_code
            }
            
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        return {
            "success": False,
            "response_time": response_time,
            "error": str(e),
            "status_code": None
        }


async def test_caching_performance():
    """Test caching performance with multiple requests to the same PDF."""
    print("🚀 Testing Latency Optimization - Document Caching")
    print("=" * 60)
    
    async with httpx.AsyncClient() as session:
        results = []
        
        # Test with increasing number of questions to show concurrent processing benefits
        test_scenarios = [
            ("Single Question", TEST_QUESTIONS[:1]),
            ("Three Questions", TEST_QUESTIONS[:3]),
            ("Five Questions", TEST_QUESTIONS[:5])
        ]
        
        for scenario_name, questions in test_scenarios:
            print(f"\n📋 Testing Scenario: {scenario_name}")
            print("-" * 40)
            
            scenario_results = []
            
            # Make 3 requests to show caching effect
            for i in range(3):
                print(f"Request {i+1}/3: ", end="", flush=True)
                
                result = await make_api_request(session, questions)
                scenario_results.append(result)
                
                if result["success"]:
                    print(f"✅ {result['response_time']:.2f}s ({result['answers_count']} answers)")
                else:
                    print(f"❌ {result['response_time']:.2f}s - Error: {result['error'][:50]}...")
                
                # Small delay between requests
                await asyncio.sleep(1)
            
            results.append((scenario_name, scenario_results))
    
    # Print summary
    print("\n📊 Performance Summary")
    print("=" * 60)
    
    for scenario_name, scenario_results in results:
        print(f"\n{scenario_name}:")
        
        successful_results = [r for r in scenario_results if r["success"]]
        if len(successful_results) >= 2:
            first_request = successful_results[0]["response_time"]
            subsequent_avg = sum(r["response_time"] for r in successful_results[1:]) / len(successful_results[1:])
            
            improvement = ((first_request - subsequent_avg) / first_request) * 100
            
            print(f"  First request (cache miss):  {first_request:.2f}s")
            print(f"  Subsequent avg (cache hit):  {subsequent_avg:.2f}s")
            print(f"  Performance improvement:     {improvement:.1f}%")
            
            if improvement > 80:
                print("  Status: 🎉 Excellent caching performance!")
            elif improvement > 50:
                print("  Status: ✅ Good caching performance")
            elif improvement > 20:
                print("  Status: ⚠️  Moderate improvement")
            else:
                print("  Status: ❌ Caching may not be working optimally")
        else:
            print("  Status: ❌ Insufficient successful requests for analysis")


async def test_concurrent_processing():
    """Test concurrent question processing performance."""
    print("\n🔄 Testing Concurrent Question Processing")
    print("=" * 60)
    
    async with httpx.AsyncClient() as session:
        # Test single question vs multiple questions to show concurrency benefits
        single_question_times = []
        multiple_question_time = None
        
        # Test single questions (simulating sequential processing)
        print("Testing individual questions (sequential simulation):")
        for i, question in enumerate(TEST_QUESTIONS[:3]):
            print(f"Question {i+1}: ", end="", flush=True)
            result = await make_api_request(session, [question])
            
            if result["success"]:
                single_question_times.append(result["response_time"])
                print(f"✅ {result['response_time']:.2f}s")
            else:
                print(f"❌ {result['response_time']:.2f}s - Error")
        
        # Test multiple questions (concurrent processing)
        print("\nTesting multiple questions (concurrent processing):")
        result = await make_api_request(session, TEST_QUESTIONS[:3])
        
        if result["success"]:
            multiple_question_time = result["response_time"]
            print(f"✅ {multiple_question_time:.2f}s for 3 questions")
        else:
            print(f"❌ {result['response_time']:.2f}s - Error")
        
        # Calculate efficiency
        if single_question_times and multiple_question_time:
            sequential_total = sum(single_question_times)
            concurrent_time = multiple_question_time
            
            print(f"\n📈 Concurrency Analysis:")
            print(f"Sequential total (estimated): {sequential_total:.2f}s")
            print(f"Concurrent processing:        {concurrent_time:.2f}s")
            
            if concurrent_time < sequential_total:
                improvement = ((sequential_total - concurrent_time) / sequential_total) * 100
                print(f"Concurrency improvement:      {improvement:.1f}%")
                print("Status: ✅ Concurrent processing is working!")
            else:
                print("Status: ⚠️  Concurrent processing may need optimization")


async def main():
    """Main test function."""
    print("🧪 LLM Query Retrieval System - Latency Optimization Test")
    print("=" * 70)
    print(f"API URL: {API_BASE_URL}")
    print(f"Test PDF: {TEST_PDF_URL}")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    
    try:
        # Test document caching
        await test_caching_performance()
        
        # Test concurrent processing
        await test_concurrent_processing()
        
        print("\n🎯 Test Recommendations:")
        print("1. For repeated PDFs: Expect 80-95% latency reduction on cache hits")
        print("2. For multiple questions: Concurrent processing should be faster")
        print("3. Monitor cache hit rates in production for optimal performance")
        print("\n✨ Testing completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    # Update these values before running
    if AUTH_TOKEN == "your-auth-token-here":
        print("❌ Please update AUTH_TOKEN in the script before running")
        exit(1)
    
    if TEST_PDF_URL == "https://example.com/test-document.pdf":
        print("❌ Please update TEST_PDF_URL with a real PDF URL before running")
        exit(1)
    
    asyncio.run(main())