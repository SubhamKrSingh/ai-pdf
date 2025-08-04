#!/usr/bin/env python3
"""
Script to fix Pinecone index dimension mismatch.

This script will delete the existing index with wrong dimension (1024)
and recreate it with the correct dimension (2048) for Jina embeddings v4.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.data.vector_store import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fix_pinecone_dimension():
    """Fix the Pinecone index dimension mismatch."""
    try:
        settings = get_settings()
        
        # Initialize Pinecone client directly
        client = Pinecone(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment
        )
        
        index_name = settings.pinecone_index_name
        logger.info(f"Checking index: {index_name}")
        
        # Check if index exists
        existing_indexes = client.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if index_name in index_names:
            # Get index info
            index_info = client.describe_index(index_name)
            current_dimension = index_info.dimension
            
            logger.info(f"Current index dimension: {current_dimension}")
            
            if current_dimension == 1024:
                logger.warning("Index has incorrect dimension (1024). Fixing...")
                
                # Delete the existing index
                logger.info(f"Deleting existing index: {index_name}")
                client.delete_index(index_name)
                
                # Wait for deletion to complete
                logger.info("Waiting for index deletion to complete...")
                await asyncio.sleep(10)
                
                # Create new index with correct dimension
                logger.info(f"Creating new index with dimension 2048: {index_name}")
                client.create_index(
                    name=index_name,
                    dimension=2048,  # Correct dimension for Jina embeddings v4
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                logger.info("Waiting for new index to be ready...")
                await asyncio.sleep(30)
                
                logger.info("✅ Index recreated successfully with correct dimension (2048)")
                
            elif current_dimension == 2048:
                logger.info("✅ Index already has correct dimension (2048)")
                
            else:
                logger.error(f"❌ Unexpected index dimension: {current_dimension}")
                return False
                
        else:
            logger.info(f"Index {index_name} doesn't exist. It will be created automatically with correct dimension.")
        
        # Test the vector store
        logger.info("Testing vector store connection...")
        vector_store = PineconeVectorStore()
        stats = await vector_store.get_index_stats()
        logger.info(f"✅ Vector store test successful. Index stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix Pinecone dimension: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(fix_pinecone_dimension())
    if success:
        print("\n🎉 Pinecone dimension fix completed successfully!")
        print("You can now run your application without the dimension mismatch error.")
    else:
        print("\n❌ Failed to fix Pinecone dimension. Check the logs above.")
        sys.exit(1)