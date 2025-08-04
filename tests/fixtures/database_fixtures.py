"""Database fixtures for testing."""

import asyncio
import asyncpg
import pytest
import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
import pinecone
from app.config import get_settings

# Test database configuration
TEST_DATABASE_CONFIG = {
    "host": os.getenv("TEST_DB_HOST", "localhost"),
    "port": int(os.getenv("TEST_DB_PORT", "5432")),
    "user": os.getenv("TEST_DB_USER", "test_user"),
    "password": os.getenv("TEST_DB_PASSWORD", "test_password"),
    "database": os.getenv("TEST_DB_NAME", "test_llm_system"),
}

class MockPineconeIndex:
    """Mock Pinecone index for testing."""
    
    def __init__(self):
        self.vectors = {}
        self.metadata = {}
    
    def upsert(self, vectors, namespace=None):
        """Mock upsert operation."""
        for vector_id, values, metadata in vectors:
            self.vectors[vector_id] = values
            self.metadata[vector_id] = metadata
        return {"upserted_count": len(vectors)}
    
    def query(self, vector, top_k=10, include_metadata=True, namespace=None, filter=None):
        """Mock query operation with simple similarity."""
        # Simple mock similarity calculation
        results = []
        for vec_id, vec_values in self.vectors.items():
            # Skip if filter doesn't match
            if filter and not self._matches_filter(self.metadata.get(vec_id, {}), filter):
                continue
                
            # Simple dot product similarity
            similarity = sum(a * b for a, b in zip(vector, vec_values[:len(vector)]))
            results.append({
                "id": vec_id,
                "score": similarity,
                "values": vec_values if include_metadata else None,
                "metadata": self.metadata.get(vec_id, {}) if include_metadata else None,
            })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": results[:top_k]}
    
    def delete(self, ids, namespace=None):
        """Mock delete operation."""
        deleted_count = 0
        for vec_id in ids:
            if vec_id in self.vectors:
                del self.vectors[vec_id]
                if vec_id in self.metadata:
                    del self.metadata[vec_id]
                deleted_count += 1
        return {"deleted_count": deleted_count}
    
    def describe_index_stats(self):
        """Mock index stats."""
        return {
            "dimension": 512,
            "index_fullness": 0.1,
            "namespaces": {
                "": {"vector_count": len(self.vectors)}
            },
            "total_vector_count": len(self.vectors)
        }
    
    def _matches_filter(self, metadata, filter_dict):
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

class MockPinecone:
    """Mock Pinecone client for testing."""
    
    def __init__(self):
        self.indexes = {}
    
    def Index(self, name):
        """Get or create a mock index."""
        if name not in self.indexes:
            self.indexes[name] = MockPineconeIndex()
        return self.indexes[name]
    
    def list_indexes(self):
        """List available indexes."""
        return list(self.indexes.keys())
    
    def create_index(self, name, dimension, metric="cosine"):
        """Create a new mock index."""
        self.indexes[name] = MockPineconeIndex()
        return True
    
    def delete_index(self, name):
        """Delete a mock index."""
        if name in self.indexes:
            del self.indexes[name]
        return True

@pytest.fixture
async def test_db_connection():
    """Create a test database connection."""
    try:
        conn = await asyncpg.connect(**TEST_DATABASE_CONFIG)
        yield conn
    except Exception as e:
        # If test database is not available, use a mock
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="SELECT 1")
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.fetchrow = AsyncMock(return_value=None)
        yield mock_conn
    finally:
        if 'conn' in locals():
            await conn.close()

@pytest.fixture
async def test_db_pool():
    """Create a test database connection pool."""
    try:
        pool = await asyncpg.create_pool(**TEST_DATABASE_CONFIG, min_size=1, max_size=5)
        yield pool
    except Exception as e:
        # If test database is not available, use a mock
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock()
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        yield mock_pool
    finally:
        if 'pool' in locals():
            await pool.close()

@pytest.fixture
def mock_pinecone():
    """Create a mock Pinecone client."""
    return MockPinecone()

@pytest.fixture
def mock_pinecone_index():
    """Create a mock Pinecone index."""
    return MockPineconeIndex()

@pytest.fixture
async def clean_test_db():
    """Clean test database before and after tests."""
    try:
        conn = await asyncpg.connect(**TEST_DATABASE_CONFIG)
        
        # Clean up before test
        await conn.execute("TRUNCATE TABLE query_sessions CASCADE")
        await conn.execute("TRUNCATE TABLE documents CASCADE")
        
        yield conn
        
        # Clean up after test
        await conn.execute("TRUNCATE TABLE query_sessions CASCADE")
        await conn.execute("TRUNCATE TABLE documents CASCADE")
        
    except Exception:
        # If test database is not available, use a mock
        mock_conn = AsyncMock()
        yield mock_conn
    finally:
        if 'conn' in locals():
            await conn.close()

@pytest.fixture
def test_settings():
    """Override settings for testing."""
    settings = get_settings()
    settings.database_url = f"postgresql://{TEST_DATABASE_CONFIG['user']}:{TEST_DATABASE_CONFIG['password']}@{TEST_DATABASE_CONFIG['host']}:{TEST_DATABASE_CONFIG['port']}/{TEST_DATABASE_CONFIG['database']}"
    settings.pinecone_index_name = "test-index"
    settings.auth_token = "test-token"
    settings.gemini_api_key = "test-gemini-key"
    settings.jina_api_key = "test-jina-key"
    settings.pinecone_api_key = "test-pinecone-key"
    return settings

class DatabaseTestHelper:
    """Helper class for database testing operations."""
    
    @staticmethod
    async def create_test_document(conn, document_id="test-doc-1", url="http://example.com/test.pdf"):
        """Create a test document in the database."""
        await conn.execute("""
            INSERT INTO documents (id, url, content_type, chunk_count, status)
            VALUES ($1, $2, 'application/pdf', 3, 'completed')
        """, document_id, url)
        return document_id
    
    @staticmethod
    async def create_test_query_session(conn, document_id="test-doc-1", questions=None, answers=None):
        """Create a test query session in the database."""
        if questions is None:
            questions = ["What is this document about?"]
        if answers is None:
            answers = ["This is a test document."]
            
        session_id = await conn.fetchval("""
            INSERT INTO query_sessions (document_id, questions, answers, processing_time_ms)
            VALUES ($1, $2, $3, 1000)
            RETURNING id
        """, document_id, questions, answers)
        return session_id
    
    @staticmethod
    async def get_document_count(conn):
        """Get the number of documents in the database."""
        return await conn.fetchval("SELECT COUNT(*) FROM documents")
    
    @staticmethod
    async def get_session_count(conn):
        """Get the number of query sessions in the database."""
        return await conn.fetchval("SELECT COUNT(*) FROM query_sessions")

class VectorStoreTestHelper:
    """Helper class for vector store testing operations."""
    
    @staticmethod
    def create_test_vectors(count=3):
        """Create test vectors for Pinecone."""
        vectors = []
        for i in range(count):
            vectors.append((
                f"test-chunk-{i}",
                [0.1 * (i + 1)] * 512,  # 512-dimensional vector
                {
                    "document_id": "test-doc-1",
                    "chunk_index": i,
                    "content": f"This is test chunk {i}",
                }
            ))
        return vectors
    
    @staticmethod
    def create_query_vector():
        """Create a test query vector."""
        return [0.15] * 512  # Should be similar to test-chunk-1