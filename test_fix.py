#!/usr/bin/env python3
"""
Quick test script to verify the embedding service fix.
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.embedding_service import get_embedding_service
from app.config import get_settings

async def test_embedding_service():
    """Test the embedding service with a simple text."""
    try:
        print("Testing embedding service...")
        
        # Get the embedding service
        service = await get_embedding_service()
        
        # Test with a simple text
        test_text = "This is a test document for embedding generation."
        
        print(f"Generating embedding for: {test_text}")
        embedding = await service.generate_query_embedding(test_text)
        
        print(f"Successfully generated embedding with {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"Error testing embedding service: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_embedding_service())
    if success:
        print("✅ Embedding service test passed!")
    else:
        print("❌ Embedding service test failed!")
        sys.exit(1)