"""
Unit tests for PostgreSQL database repository.
Tests repository logic with mocked database connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from app.data.repository import DatabaseRepository, DatabaseError
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.database_url = "postgresql://test:test@localhost:5432/test"
    return settings


@pytest.fixture
def repository(mock_settings):
    """Create repository instance with mocked settings."""
    with patch('app.data.repository.get_settings', return_value=mock_settings):
        repo = DatabaseRepository()
        return repo


@pytest.fixture
def mock_connection():
    """Mock database connection."""
    conn = AsyncMock()
    return conn


@pytest.fixture
def mock_pool():
    """Mock connection pool."""
    pool = AsyncMock()
    # Use MagicMock for synchronous methods
    pool.get_size = MagicMock(return_value=10)
    pool.get_min_size = MagicMock(return_value=5)
    pool.get_max_size = MagicMock(return_value=20)
    pool.get_idle_size = MagicMock(return_value=3)
    return pool


class TestDatabaseRepository:
    """Test cases for DatabaseRepository class."""
    
    async def test_initialization_success(self, repository, mock_pool):
        """Test successful repository initialization."""
        async def mock_create_pool_func(*args, **kwargs):
            return mock_pool
        
        with patch('asyncpg.create_pool', side_effect=mock_create_pool_func) as mock_create_pool:
            await repository.initialize()
            
            mock_create_pool.assert_called_once_with(
                repository.settings.database_url,
                min_size=5,
                max_size=20,
                command_timeout=30,
                server_settings={
                    'application_name': 'llm_query_retrieval_system',
                    'timezone': 'UTC'
                }
            )
            assert repository._pool is mock_pool
    
    async def test_initialization_failure(self, repository):
        """Test repository initialization failure."""
        with patch('asyncpg.create_pool', side_effect=Exception("Connection failed")):
            with pytest.raises(DatabaseError) as exc_info:
                await repository.initialize()
            
            assert exc_info.value.operation == "initialize"
            assert "Failed to initialize database connection pool" in exc_info.value.message
            assert "Connection failed" in str(exc_info.value.details["error"])
    
    async def test_close_repository(self, repository, mock_pool):
        """Test repository closure."""
        repository._pool = mock_pool
        
        await repository.close()
        
        mock_pool.close.assert_called_once()
        assert repository._pool is None
    
    async def test_get_connection_success(self, repository, mock_pool, mock_connection):
        """Test successful connection acquisition."""
        repository._pool = mock_pool
        mock_pool.acquire.return_value = mock_connection
        
        async with repository.get_connection() as conn:
            assert conn is mock_connection
        
        mock_pool.acquire.assert_called_once()
        mock_pool.release.assert_called_once_with(mock_connection)
    
    async def test_get_connection_failure(self, repository, mock_pool):
        """Test connection acquisition failure."""
        repository._pool = mock_pool
        mock_pool.acquire.side_effect = Exception("Pool exhausted")
        
        with pytest.raises(DatabaseError) as exc_info:
            async with repository.get_connection():
                pass
        
        assert exc_info.value.operation == "get_connection"
        assert "Database connection error" in exc_info.value.message
    
    async def test_store_document_metadata_success(self, repository, mock_connection):
        """Test successful document metadata storage."""
        document_id = str(uuid4())
        url = "https://example.com/test.pdf"
        content_type = "application/pdf"
        chunk_count = 5
        status = "completed"
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            result = await repository.store_document_metadata(
                document_id, url, content_type, chunk_count, status
            )
            
            assert result is True
            mock_connection.execute.assert_called_once()
            
            # Verify the SQL call
            call_args = mock_connection.execute.call_args
            assert "INSERT INTO documents" in call_args[0][0]
            assert call_args[0][1] == document_id
            assert call_args[0][2] == url
            assert call_args[0][3] == content_type
            assert call_args[0][4] == chunk_count
            assert call_args[0][5] == status
    
    async def test_store_document_metadata_failure(self, repository, mock_connection):
        """Test document metadata storage failure."""
        mock_connection.execute.side_effect = Exception("Database error")
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            with pytest.raises(DatabaseError) as exc_info:
                await repository.store_document_metadata(
                    "test-id", "test-url", "application/pdf", 5
                )
            
            assert exc_info.value.operation == "store_document_metadata"
            assert "Failed to store document metadata" in exc_info.value.message
    
    async def test_get_document_metadata_success(self, repository, mock_connection):
        """Test successful document metadata retrieval."""
        document_id = str(uuid4())
        mock_row = {
            "id": document_id,
            "url": "https://example.com/test.pdf",
            "content_type": "application/pdf",
            "processed_at": datetime.utcnow(),
            "chunk_count": 5,
            "status": "completed"
        }
        mock_connection.fetchrow.return_value = mock_row
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            result = await repository.get_document_metadata(document_id)
            
            assert result is not None
            assert result["id"] == document_id
            assert result["url"] == mock_row["url"]
            assert result["content_type"] == mock_row["content_type"]
            assert result["chunk_count"] == mock_row["chunk_count"]
            assert result["status"] == mock_row["status"]
            assert "processed_at" in result
    
    async def test_get_document_metadata_not_found(self, repository, mock_connection):
        """Test document metadata retrieval when document not found."""
        mock_connection.fetchrow.return_value = None
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            result = await repository.get_document_metadata("nonexistent-id")
            
            assert result is None
    
    async def test_get_document_metadata_failure(self, repository, mock_connection):
        """Test document metadata retrieval failure."""
        mock_connection.fetchrow.side_effect = Exception("Database error")
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            with pytest.raises(DatabaseError) as exc_info:
                await repository.get_document_metadata("test-id")
            
            assert exc_info.value.operation == "get_document_metadata"
            assert "Failed to retrieve document metadata" in exc_info.value.message
    
    async def test_log_query_session_success(self, repository, mock_connection):
        """Test successful query session logging."""
        document_id = str(uuid4())
        questions = ["What is this about?", "Who wrote it?"]
        answers = ["It's about AI", "John Doe"]
        processing_time = 1500
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            session_id = await repository.log_query_session(
                document_id, questions, answers, processing_time
            )
            
            assert session_id is not None
            assert len(session_id) > 0
            mock_connection.execute.assert_called_once()
            
            # Verify the SQL call
            call_args = mock_connection.execute.call_args
            assert "INSERT INTO query_sessions" in call_args[0][0]
            assert call_args[0][2] == document_id
            assert call_args[0][3] == questions
            assert call_args[0][4] == answers
            assert call_args[0][5] == processing_time
    
    async def test_log_query_session_failure(self, repository, mock_connection):
        """Test query session logging failure."""
        mock_connection.execute.side_effect = Exception("Database error")
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            with pytest.raises(DatabaseError) as exc_info:
                await repository.log_query_session(
                    "test-id", ["question"], ["answer"], 1000
                )
            
            assert exc_info.value.operation == "log_query_session"
            assert "Failed to log query session" in exc_info.value.message
    
    async def test_get_query_sessions_with_document_filter(self, repository, mock_connection):
        """Test query session retrieval with document filter."""
        document_id = str(uuid4())
        session_id = str(uuid4())
        mock_rows = [{
            "id": session_id,
            "document_id": document_id,
            "questions": ["What is this?"],
            "answers": ["It's a test"],
            "processing_time_ms": 1000,
            "created_at": datetime.utcnow()
        }]
        mock_connection.fetch.return_value = mock_rows
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            result = await repository.get_query_sessions(document_id=document_id)
            
            assert len(result) == 1
            assert result[0]["id"] == session_id
            assert result[0]["document_id"] == document_id
            assert result[0]["questions"] == ["What is this?"]
            assert result[0]["answers"] == ["It's a test"]
    
    async def test_get_query_sessions_without_filter(self, repository, mock_connection):
        """Test query session retrieval without document filter."""
        mock_rows = []
        mock_connection.fetch.return_value = mock_rows
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            result = await repository.get_query_sessions(limit=50)
            
            assert result == []
            
            # Verify correct SQL was called (without document_id filter)
            call_args = mock_connection.fetch.call_args
            assert "WHERE document_id" not in call_args[0][0]
    
    async def test_delete_document_success(self, repository, mock_connection):
        """Test successful document deletion."""
        document_id = str(uuid4())
        mock_connection.execute.side_effect = [None, "DELETE 1"]  # Sessions delete, then document delete
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_transaction)
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            result = await repository.delete_document(document_id)
            
            assert result is True
            assert mock_connection.execute.call_count == 2
    
    async def test_delete_document_not_found(self, repository, mock_connection):
        """Test document deletion when document not found."""
        document_id = str(uuid4())
        mock_connection.execute.side_effect = [None, "DELETE 0"]  # No document deleted
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_transaction)
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            result = await repository.delete_document(document_id)
            
            assert result is False
    
    async def test_delete_document_failure(self, repository, mock_connection):
        """Test document deletion failure."""
        mock_connection.execute.side_effect = Exception("Database error")
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_connection.transaction = MagicMock(return_value=mock_transaction)
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            with pytest.raises(DatabaseError) as exc_info:
                await repository.delete_document("test-id")
            
            assert exc_info.value.operation == "delete_document"
            assert "Failed to delete document" in exc_info.value.message
    
    async def test_health_check_success(self, repository, mock_connection, mock_pool):
        """Test successful health check."""
        repository._pool = mock_pool
        mock_connection.fetchval.return_value = 1
        

        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            result = await repository.health_check()
            
            assert result["status"] == "healthy"
            assert result["connection_test"] is True
            assert "pool_stats" in result
            assert result["pool_stats"]["size"] == 10
            assert result["pool_stats"]["min_size"] == 5
            assert result["pool_stats"]["max_size"] == 20
            assert result["pool_stats"]["idle_size"] == 3
    
    async def test_health_check_failure(self, repository, mock_connection):
        """Test health check failure."""
        mock_connection.fetchval.side_effect = Exception("Connection failed")
        
        with patch.object(repository, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            
            with pytest.raises(DatabaseError) as exc_info:
                await repository.health_check()
            
            assert exc_info.value.operation == "health_check"
            assert "Database health check failed" in exc_info.value.message


class TestDatabaseError:
    """Test cases for DatabaseError exception."""
    
    def test_database_error_creation(self):
        """Test DatabaseError exception creation."""
        error = DatabaseError(
            message="Test error",
            operation="test_operation",
            details={"key": "value"}
        )
        
        assert error.message == "Test error"
        assert error.operation == "test_operation"
        assert error.details == {"key": "value"}
        assert str(error) == "Test error"
    
    def test_database_error_without_details(self):
        """Test DatabaseError exception creation without details."""
        error = DatabaseError(
            message="Test error",
            operation="test_operation"
        )
        
        assert error.message == "Test error"
        assert error.operation == "test_operation"
        assert error.details == {}


class TestGlobalRepositoryFunctions:
    """Test cases for global repository functions."""
    
    @patch('app.data.repository._repository', None)
    async def test_get_repository_creates_new_instance(self, mock_settings):
        """Test that get_repository creates new instance when none exists."""
        with patch('app.data.repository.get_settings', return_value=mock_settings):
            with patch('app.data.repository.DatabaseRepository') as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                
                from app.data.repository import get_repository
                
                result = await get_repository()
                
                assert result is mock_repo
                mock_repo.initialize.assert_called_once()
    
    @patch('app.data.repository._repository')
    async def test_get_repository_returns_existing_instance(self, mock_existing_repo):
        """Test that get_repository returns existing instance."""
        from app.data.repository import get_repository
        
        result = await get_repository()
        
        assert result is mock_existing_repo
        mock_existing_repo.initialize.assert_not_called()
    
    @patch('app.data.repository._repository')
    async def test_close_repository(self, mock_existing_repo):
        """Test closing global repository."""
        # Make the close method async
        mock_existing_repo.close = AsyncMock()
        
        from app.data.repository import close_repository
        
        await close_repository()
        
        mock_existing_repo.close.assert_called_once()