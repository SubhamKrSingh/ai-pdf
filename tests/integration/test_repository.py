"""
Integration tests for PostgreSQL database repository.
Tests database operations with real database connections.
"""

import pytest
import asyncio
import os
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any

from app.data.repository import DatabaseRepository, DatabaseError, get_repository, close_repository
from app.data.migrations import run_migrations, get_migration_status
from app.config import get_settings


# Test database configuration
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://test:test@localhost:5432/test_llm_query")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def setup_test_database():
    """Set up test database with migrations."""
    # Override database URL for testing
    original_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    
    try:
        # Run migrations
        applied_migrations = await run_migrations()
        assert len(applied_migrations) > 0, "No migrations were applied"
        
        yield
        
    finally:
        # Restore original database URL
        if original_url:
            os.environ["DATABASE_URL"] = original_url
        else:
            os.environ.pop("DATABASE_URL", None)
        
        # Close repository connections
        await close_repository()


@pytest.fixture
async def repository(setup_test_database):
    """Get database repository instance for testing."""
    repo = DatabaseRepository()
    await repo.initialize()
    yield repo
    await repo.close()


@pytest.fixture
async def sample_document_data():
    """Sample document data for testing."""
    return {
        "document_id": str(uuid4()),
        "url": "https://example.com/test-document.pdf",
        "content_type": "application/pdf",
        "chunk_count": 5,
        "status": "completed"
    }


@pytest.fixture
async def sample_query_data():
    """Sample query session data for testing."""
    return {
        "questions": ["What is the main topic?", "Who are the key stakeholders?"],
        "answers": ["The main topic is AI development.", "The key stakeholders are developers and users."],
        "processing_time_ms": 1500
    }


class TestDatabaseRepository:
    """Test cases for DatabaseRepository class."""
    
    async def test_repository_initialization(self, repository):
        """Test repository initialization and connection pool setup."""
        assert repository._pool is not None
        assert repository._pool.get_size() > 0
        
        # Test health check
        health = await repository.health_check()
        assert health["status"] == "healthy"
        assert health["connection_test"] is True
        assert "pool_stats" in health
    
    async def test_store_and_retrieve_document_metadata(self, repository, sample_document_data):
        """Test storing and retrieving document metadata."""
        # Store document metadata
        success = await repository.store_document_metadata(
            document_id=sample_document_data["document_id"],
            url=sample_document_data["url"],
            content_type=sample_document_data["content_type"],
            chunk_count=sample_document_data["chunk_count"],
            status=sample_document_data["status"]
        )
        assert success is True
        
        # Retrieve document metadata
        metadata = await repository.get_document_metadata(sample_document_data["document_id"])
        assert metadata is not None
        assert metadata["id"] == sample_document_data["document_id"]
        assert metadata["url"] == sample_document_data["url"]
        assert metadata["content_type"] == sample_document_data["content_type"]
        assert metadata["chunk_count"] == sample_document_data["chunk_count"]
        assert metadata["status"] == sample_document_data["status"]
        assert "processed_at" in metadata
    
    async def test_retrieve_nonexistent_document(self, repository):
        """Test retrieving metadata for non-existent document."""
        nonexistent_id = str(uuid4())
        metadata = await repository.get_document_metadata(nonexistent_id)
        assert metadata is None
    
    async def test_update_document_metadata(self, repository, sample_document_data):
        """Test updating existing document metadata."""
        # Store initial metadata
        await repository.store_document_metadata(
            document_id=sample_document_data["document_id"],
            url=sample_document_data["url"],
            content_type=sample_document_data["content_type"],
            chunk_count=sample_document_data["chunk_count"]
        )
        
        # Update metadata
        updated_chunk_count = 10
        updated_status = "updated"
        success = await repository.store_document_metadata(
            document_id=sample_document_data["document_id"],
            url=sample_document_data["url"],
            content_type=sample_document_data["content_type"],
            chunk_count=updated_chunk_count,
            status=updated_status
        )
        assert success is True
        
        # Verify update
        metadata = await repository.get_document_metadata(sample_document_data["document_id"])
        assert metadata["chunk_count"] == updated_chunk_count
        assert metadata["status"] == updated_status
    
    async def test_log_and_retrieve_query_sessions(self, repository, sample_document_data, sample_query_data):
        """Test logging and retrieving query sessions."""
        # Store document first
        await repository.store_document_metadata(
            document_id=sample_document_data["document_id"],
            url=sample_document_data["url"],
            content_type=sample_document_data["content_type"],
            chunk_count=sample_document_data["chunk_count"]
        )
        
        # Log query session
        session_id = await repository.log_query_session(
            document_id=sample_document_data["document_id"],
            questions=sample_query_data["questions"],
            answers=sample_query_data["answers"],
            processing_time_ms=sample_query_data["processing_time_ms"]
        )
        assert session_id is not None
        assert len(session_id) > 0
        
        # Retrieve query sessions
        sessions = await repository.get_query_sessions(
            document_id=sample_document_data["document_id"]
        )
        assert len(sessions) == 1
        
        session = sessions[0]
        assert session["id"] == session_id
        assert session["document_id"] == sample_document_data["document_id"]
        assert session["questions"] == sample_query_data["questions"]
        assert session["answers"] == sample_query_data["answers"]
        assert session["processing_time_ms"] == sample_query_data["processing_time_ms"]
        assert "created_at" in session
    
    async def test_retrieve_all_query_sessions(self, repository, sample_document_data, sample_query_data):
        """Test retrieving all query sessions without document filter."""
        # Store document
        await repository.store_document_metadata(
            document_id=sample_document_data["document_id"],
            url=sample_document_data["url"],
            content_type=sample_document_data["content_type"],
            chunk_count=sample_document_data["chunk_count"]
        )
        
        # Log multiple query sessions
        session_ids = []
        for i in range(3):
            session_id = await repository.log_query_session(
                document_id=sample_document_data["document_id"],
                questions=[f"Question {i}"],
                answers=[f"Answer {i}"],
                processing_time_ms=1000 + i * 100
            )
            session_ids.append(session_id)
        
        # Retrieve all sessions
        sessions = await repository.get_query_sessions(limit=10)
        assert len(sessions) >= 3
        
        # Check that our sessions are included
        retrieved_ids = [session["id"] for session in sessions]
        for session_id in session_ids:
            assert session_id in retrieved_ids
    
    async def test_delete_document_and_sessions(self, repository, sample_document_data, sample_query_data):
        """Test deleting document and associated query sessions."""
        # Store document and log session
        await repository.store_document_metadata(
            document_id=sample_document_data["document_id"],
            url=sample_document_data["url"],
            content_type=sample_document_data["content_type"],
            chunk_count=sample_document_data["chunk_count"]
        )
        
        await repository.log_query_session(
            document_id=sample_document_data["document_id"],
            questions=sample_query_data["questions"],
            answers=sample_query_data["answers"],
            processing_time_ms=sample_query_data["processing_time_ms"]
        )
        
        # Verify data exists
        metadata = await repository.get_document_metadata(sample_document_data["document_id"])
        assert metadata is not None
        
        sessions = await repository.get_query_sessions(sample_document_data["document_id"])
        assert len(sessions) == 1
        
        # Delete document
        success = await repository.delete_document(sample_document_data["document_id"])
        assert success is True
        
        # Verify deletion
        metadata = await repository.get_document_metadata(sample_document_data["document_id"])
        assert metadata is None
        
        sessions = await repository.get_query_sessions(sample_document_data["document_id"])
        assert len(sessions) == 0
    
    async def test_delete_nonexistent_document(self, repository):
        """Test deleting non-existent document."""
        nonexistent_id = str(uuid4())
        success = await repository.delete_document(nonexistent_id)
        assert success is False
    
    async def test_connection_error_handling(self):
        """Test error handling for connection failures."""
        # Create repository with invalid database URL
        original_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = "postgresql://invalid:invalid@localhost:9999/invalid"
        
        try:
            repo = DatabaseRepository()
            
            with pytest.raises(DatabaseError) as exc_info:
                await repo.initialize()
            
            assert exc_info.value.operation == "initialize"
            assert "Failed to initialize database connection pool" in exc_info.value.message
            
        finally:
            if original_url:
                os.environ["DATABASE_URL"] = original_url
            else:
                os.environ.pop("DATABASE_URL", None)
    
    async def test_concurrent_operations(self, repository, sample_document_data):
        """Test concurrent database operations."""
        # Create multiple document IDs
        document_ids = [str(uuid4()) for _ in range(5)]
        
        # Store documents concurrently
        tasks = []
        for i, doc_id in enumerate(document_ids):
            task = repository.store_document_metadata(
                document_id=doc_id,
                url=f"https://example.com/doc-{i}.pdf",
                content_type="application/pdf",
                chunk_count=i + 1
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        assert all(results), "Not all concurrent stores succeeded"
        
        # Retrieve documents concurrently
        retrieve_tasks = [
            repository.get_document_metadata(doc_id) 
            for doc_id in document_ids
        ]
        
        metadata_list = await asyncio.gather(*retrieve_tasks)
        assert len(metadata_list) == 5
        assert all(metadata is not None for metadata in metadata_list)
        
        # Verify each document
        for i, metadata in enumerate(metadata_list):
            assert metadata["chunk_count"] == i + 1
            assert f"doc-{i}.pdf" in metadata["url"]


class TestGlobalRepository:
    """Test cases for global repository functions."""
    
    async def test_get_global_repository(self, setup_test_database):
        """Test getting global repository instance."""
        repo1 = await get_repository()
        repo2 = await get_repository()
        
        # Should return the same instance
        assert repo1 is repo2
        assert repo1._pool is not None
        
        # Test health check
        health = await repo1.health_check()
        assert health["status"] == "healthy"
    
    async def test_close_global_repository(self, setup_test_database):
        """Test closing global repository."""
        repo = await get_repository()
        assert repo._pool is not None
        
        await close_repository()
        assert repo._pool is None
        
        # Getting repository again should create new instance
        new_repo = await get_repository()
        assert new_repo._pool is not None


class TestMigrationStatus:
    """Test cases for migration status checking."""
    
    async def test_migration_status(self, setup_test_database):
        """Test getting migration status."""
        status = await get_migration_status()
        assert isinstance(status, list)
        assert len(status) > 0
        
        # Check that required migrations are present
        migration_names = [migration["migration_name"] for migration in status]
        expected_migrations = [
            "000_create_extensions",
            "001_create_documents_table", 
            "002_create_query_sessions_table"
        ]
        
        for expected in expected_migrations:
            assert expected in migration_names, f"Migration {expected} not found"
        
        # Check migration structure
        for migration in status:
            assert "migration_name" in migration
            assert "applied_at" in migration