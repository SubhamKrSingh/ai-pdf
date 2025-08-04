#!/usr/bin/env python3
"""
Simple script to test database repository and migrations functionality.
This script can be used to verify the database setup works correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data.migrations import run_migrations, get_migration_status
from app.data.repository import get_repository, close_repository
from app.config import get_settings


async def test_migrations():
    """Test database migrations."""
    print("🔄 Testing database migrations...")
    
    try:
        # Run migrations
        applied_migrations = await run_migrations()
        print(f"✅ Successfully applied {len(applied_migrations)} migrations:")
        for migration in applied_migrations:
            print(f"   - {migration}")
        
        # Get migration status
        status = await get_migration_status()
        print(f"\n📊 Migration status ({len(status)} migrations):")
        for migration in status:
            if "status" in migration and migration["status"] == "error":
                print(f"   ❌ Error: {migration.get('error', 'Unknown error')}")
            else:
                print(f"   ✅ {migration['migration_name']} - {migration['applied_at']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration test failed: {str(e)}")
        return False


async def test_repository():
    """Test database repository functionality."""
    print("\n🔄 Testing database repository...")
    
    try:
        # Get repository instance
        repo = await get_repository()
        
        # Test health check
        health = await repo.health_check()
        print(f"✅ Database health check: {health['status']}")
        print(f"   Connection test: {health['connection_test']}")
        print(f"   Pool size: {health['pool_stats']['size']}")
        
        # Test document metadata operations
        document_id = "test-doc-123"
        url = "https://example.com/test.pdf"
        content_type = "application/pdf"
        chunk_count = 5
        
        # Store document metadata
        success = await repo.store_document_metadata(
            document_id=document_id,
            url=url,
            content_type=content_type,
            chunk_count=chunk_count
        )
        print(f"✅ Store document metadata: {success}")
        
        # Retrieve document metadata
        metadata = await repo.get_document_metadata(document_id)
        if metadata:
            print(f"✅ Retrieved document metadata: {metadata['id']}")
            print(f"   URL: {metadata['url']}")
            print(f"   Content type: {metadata['content_type']}")
            print(f"   Chunk count: {metadata['chunk_count']}")
        else:
            print("❌ Failed to retrieve document metadata")
            return False
        
        # Test query session logging
        questions = ["What is this document about?", "Who is the author?"]
        answers = ["This is a test document", "Test Author"]
        processing_time = 1500
        
        session_id = await repo.log_query_session(
            document_id=document_id,
            questions=questions,
            answers=answers,
            processing_time_ms=processing_time
        )
        print(f"✅ Logged query session: {session_id}")
        
        # Retrieve query sessions
        sessions = await repo.get_query_sessions(document_id=document_id)
        print(f"✅ Retrieved {len(sessions)} query sessions")
        
        # Clean up test data
        deleted = await repo.delete_document(document_id)
        print(f"✅ Cleaned up test document: {deleted}")
        
        return True
        
    except Exception as e:
        print(f"❌ Repository test failed: {str(e)}")
        return False
    finally:
        await close_repository()


async def main():
    """Main test function."""
    print("🚀 Starting database tests...\n")
    
    # Check if required environment variables are set
    try:
        settings = get_settings()
        print(f"✅ Configuration loaded successfully")
        print(f"   Database URL: {settings.database_url[:20]}...")
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        print("\n💡 Make sure you have set the required environment variables:")
        print("   - DATABASE_URL")
        print("   - AUTH_TOKEN")
        print("   - GEMINI_API_KEY")
        print("   - JINA_API_KEY")
        print("   - PINECONE_API_KEY")
        print("   - PINECONE_ENVIRONMENT")
        return False
    
    # Run tests
    migration_success = await test_migrations()
    repository_success = await test_repository()
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   Migrations: {'✅ PASSED' if migration_success else '❌ FAILED'}")
    print(f"   Repository: {'✅ PASSED' if repository_success else '❌ FAILED'}")
    
    if migration_success and repository_success:
        print(f"\n🎉 All database tests passed!")
        return True
    else:
        print(f"\n💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)