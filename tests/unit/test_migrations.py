"""
Unit tests for database migration system.
Tests migration logic with mocked database connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.data.migrations import DatabaseMigrator, MigrationError, run_migrations, get_migration_status
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.database_url = "postgresql://test:test@localhost:5432/test"
    return settings


@pytest.fixture
def migrator(mock_settings):
    """Create migrator instance with mocked settings."""
    with patch('app.data.migrations.get_settings', return_value=mock_settings):
        return DatabaseMigrator()


@pytest.fixture
def mock_connection():
    """Mock database connection."""
    conn = AsyncMock()
    return conn


class TestDatabaseMigrator:
    """Test cases for DatabaseMigrator class."""
    
    async def test_create_connection_success(self, migrator, mock_connection):
        """Test successful database connection creation."""
        with patch('asyncpg.connect', return_value=mock_connection) as mock_connect:
            result = await migrator.create_connection()
            
            assert result is mock_connection
            mock_connect.assert_called_once_with(migrator.settings.database_url)
    
    async def test_create_connection_failure(self, migrator):
        """Test database connection creation failure."""
        with patch('asyncpg.connect', side_effect=Exception("Connection failed")):
            with pytest.raises(MigrationError) as exc_info:
                await migrator.create_connection()
            
            assert exc_info.value.migration_name == "connection"
            assert "Failed to create database connection" in exc_info.value.message
    
    async def test_create_migrations_table_success(self, migrator, mock_connection):
        """Test successful migrations table creation."""
        await migrator.create_migrations_table(mock_connection)
        
        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS schema_migrations" in call_args
    
    async def test_create_migrations_table_failure(self, migrator, mock_connection):
        """Test migrations table creation failure."""
        mock_connection.execute.side_effect = Exception("Table creation failed")
        
        with pytest.raises(MigrationError) as exc_info:
            await migrator.create_migrations_table(mock_connection)
        
        assert exc_info.value.migration_name == "schema_migrations"
        assert "Failed to create migrations table" in exc_info.value.message
    
    async def test_is_migration_applied_true(self, migrator, mock_connection):
        """Test checking if migration is applied (returns True)."""
        mock_connection.fetchval.return_value = 1
        
        result = await migrator.is_migration_applied(mock_connection, "test_migration")
        
        assert result is True
        mock_connection.fetchval.assert_called_once_with(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_name = $1",
            "test_migration"
        )
    
    async def test_is_migration_applied_false(self, migrator, mock_connection):
        """Test checking if migration is applied (returns False)."""
        mock_connection.fetchval.return_value = 0
        
        result = await migrator.is_migration_applied(mock_connection, "test_migration")
        
        assert result is False
    
    async def test_is_migration_applied_table_not_exists(self, migrator, mock_connection):
        """Test checking migration when table doesn't exist."""
        mock_connection.fetchval.side_effect = Exception("Table doesn't exist")
        
        result = await migrator.is_migration_applied(mock_connection, "test_migration")
        
        assert result is False
    
    async def test_record_migration_success(self, migrator, mock_connection):
        """Test successful migration recording."""
        await migrator.record_migration(mock_connection, "test_migration", "checksum123")
        
        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args
        assert "INSERT INTO schema_migrations" in call_args[0][0]
        assert call_args[0][1] == "test_migration"
        assert call_args[0][2] == "checksum123"
    
    async def test_record_migration_failure(self, migrator, mock_connection):
        """Test migration recording failure."""
        mock_connection.execute.side_effect = Exception("Insert failed")
        
        with pytest.raises(MigrationError) as exc_info:
            await migrator.record_migration(mock_connection, "test_migration")
        
        assert exc_info.value.migration_name == "test_migration"
        assert "Failed to record migration" in exc_info.value.message
    
    async def test_create_extensions_success(self, migrator, mock_connection):
        """Test successful extensions creation."""
        # Mock migration not applied
        with patch.object(migrator, 'is_migration_applied', return_value=False):
            with patch.object(migrator, 'record_migration') as mock_record:
                await migrator.create_extensions(mock_connection)
                
                # Should execute extension creation commands
                assert mock_connection.execute.call_count == 2
                mock_record.assert_called_once_with(mock_connection, "000_create_extensions")
    
    async def test_create_extensions_already_applied(self, migrator, mock_connection):
        """Test extensions creation when already applied."""
        with patch.object(migrator, 'is_migration_applied', return_value=True):
            await migrator.create_extensions(mock_connection)
            
            # Should not execute any commands
            mock_connection.execute.assert_not_called()
    
    async def test_create_extensions_failure(self, migrator, mock_connection):
        """Test extensions creation failure."""
        mock_connection.execute.side_effect = Exception("Extension creation failed")
        
        with patch.object(migrator, 'is_migration_applied', return_value=False):
            with pytest.raises(MigrationError) as exc_info:
                await migrator.create_extensions(mock_connection)
            
            assert exc_info.value.migration_name == "000_create_extensions"
            assert "Failed to create extensions" in exc_info.value.message
    
    async def test_create_documents_table_success(self, migrator, mock_connection):
        """Test successful documents table creation."""
        with patch.object(migrator, 'is_migration_applied', return_value=False):
            with patch.object(migrator, 'record_migration') as mock_record:
                await migrator.create_documents_table(mock_connection)
                
                mock_connection.execute.assert_called_once()
                call_args = mock_connection.execute.call_args[0][0]
                assert "CREATE TABLE documents" in call_args
                assert "CREATE INDEX idx_documents_url" in call_args
                mock_record.assert_called_once_with(mock_connection, "001_create_documents_table")
    
    async def test_create_documents_table_already_applied(self, migrator, mock_connection):
        """Test documents table creation when already applied."""
        with patch.object(migrator, 'is_migration_applied', return_value=True):
            await migrator.create_documents_table(mock_connection)
            
            mock_connection.execute.assert_not_called()
    
    async def test_create_query_sessions_table_success(self, migrator, mock_connection):
        """Test successful query sessions table creation."""
        with patch.object(migrator, 'is_migration_applied', return_value=False):
            with patch.object(migrator, 'record_migration') as mock_record:
                await migrator.create_query_sessions_table(mock_connection)
                
                mock_connection.execute.assert_called_once()
                call_args = mock_connection.execute.call_args[0][0]
                assert "CREATE TABLE query_sessions" in call_args
                assert "CREATE INDEX idx_query_sessions_document_id" in call_args
                mock_record.assert_called_once_with(mock_connection, "002_create_query_sessions_table")
    
    async def test_run_all_migrations_success(self, migrator, mock_connection):
        """Test successful execution of all migrations."""
        with patch.object(migrator, 'create_connection', return_value=mock_connection):
            with patch.object(migrator, 'create_migrations_table') as mock_create_table:
                with patch.object(migrator, 'create_extensions') as mock_extensions:
                    with patch.object(migrator, 'create_documents_table') as mock_documents:
                        with patch.object(migrator, 'create_query_sessions_table') as mock_sessions:
                            
                            result = await migrator.run_all_migrations()
                            
                            assert len(result) == 3
                            assert "000_create_extensions" in result
                            assert "001_create_documents_table" in result
                            assert "002_create_query_sessions_table" in result
                            
                            mock_create_table.assert_called_once_with(mock_connection)
                            mock_extensions.assert_called_once_with(mock_connection)
                            mock_documents.assert_called_once_with(mock_connection)
                            mock_sessions.assert_called_once_with(mock_connection)
                            mock_connection.close.assert_called_once()
    
    async def test_run_all_migrations_failure(self, migrator, mock_connection):
        """Test migration failure during execution."""
        with patch.object(migrator, 'create_connection', return_value=mock_connection):
            with patch.object(migrator, 'create_migrations_table'):
                with patch.object(migrator, 'create_extensions', side_effect=Exception("Migration failed")):
                    
                    with pytest.raises(MigrationError) as exc_info:
                        await migrator.run_all_migrations()
                    
                    assert exc_info.value.migration_name == "000_create_extensions"
                    assert "Unexpected error in migration" in exc_info.value.message
                    mock_connection.close.assert_called_once()
    
    async def test_run_all_migrations_migration_error(self, migrator, mock_connection):
        """Test migration error propagation."""
        migration_error = MigrationError("Test error", "test_migration")
        
        with patch.object(migrator, 'create_connection', return_value=mock_connection):
            with patch.object(migrator, 'create_migrations_table'):
                with patch.object(migrator, 'create_extensions', side_effect=migration_error):
                    
                    with pytest.raises(MigrationError) as exc_info:
                        await migrator.run_all_migrations()
                    
                    assert exc_info.value is migration_error
    
    async def test_get_migration_status_success(self, migrator, mock_connection):
        """Test successful migration status retrieval."""
        mock_rows = [
            {
                "migration_name": "000_create_extensions",
                "applied_at": datetime.utcnow(),
                "checksum": "abc123"
            },
            {
                "migration_name": "001_create_documents_table",
                "applied_at": datetime.utcnow(),
                "checksum": "def456"
            }
        ]
        mock_connection.fetchval.return_value = True  # Table exists
        mock_connection.fetch.return_value = mock_rows
        
        with patch.object(migrator, 'create_connection', return_value=mock_connection):
            result = await migrator.get_migration_status()
            
            assert len(result) == 2
            assert result[0]["migration_name"] == "000_create_extensions"
            assert result[1]["migration_name"] == "001_create_documents_table"
            assert "applied_at" in result[0]
            assert "checksum" in result[0]
            mock_connection.close.assert_called_once()
    
    async def test_get_migration_status_table_not_found(self, migrator, mock_connection):
        """Test migration status when migrations table doesn't exist."""
        mock_connection.fetchval.return_value = False  # Table doesn't exist
        
        with patch.object(migrator, 'create_connection', return_value=mock_connection):
            result = await migrator.get_migration_status()
            
            assert len(result) == 1
            assert result[0]["status"] == "migrations_table_not_found"
    
    async def test_get_migration_status_error(self, migrator, mock_connection):
        """Test migration status retrieval error."""
        mock_connection.fetchval.side_effect = Exception("Database error")
        
        with patch.object(migrator, 'create_connection', return_value=mock_connection):
            result = await migrator.get_migration_status()
            
            assert len(result) == 1
            assert result[0]["status"] == "error"
            assert "error" in result[0]


class TestMigrationError:
    """Test cases for MigrationError exception."""
    
    def test_migration_error_creation(self):
        """Test MigrationError exception creation."""
        error = MigrationError(
            message="Test migration error",
            migration_name="test_migration",
            details={"key": "value"}
        )
        
        assert error.message == "Test migration error"
        assert error.migration_name == "test_migration"
        assert error.details == {"key": "value"}
        assert str(error) == "Test migration error"
    
    def test_migration_error_without_details(self):
        """Test MigrationError exception creation without details."""
        error = MigrationError(
            message="Test migration error",
            migration_name="test_migration"
        )
        
        assert error.message == "Test migration error"
        assert error.migration_name == "test_migration"
        assert error.details == {}


class TestGlobalMigrationFunctions:
    """Test cases for global migration functions."""
    
    async def test_run_migrations_function(self, mock_settings):
        """Test global run_migrations function."""
        with patch('app.data.migrations.get_settings', return_value=mock_settings):
            with patch('app.data.migrations.DatabaseMigrator') as mock_migrator_class:
                mock_migrator = AsyncMock()
                mock_migrator.run_all_migrations.return_value = ["migration1", "migration2"]
                mock_migrator_class.return_value = mock_migrator
                
                result = await run_migrations()
                
                assert result == ["migration1", "migration2"]
                mock_migrator.run_all_migrations.assert_called_once()
    
    async def test_get_migration_status_function(self, mock_settings):
        """Test global get_migration_status function."""
        with patch('app.data.migrations.get_settings', return_value=mock_settings):
            with patch('app.data.migrations.DatabaseMigrator') as mock_migrator_class:
                mock_migrator = AsyncMock()
                mock_status = [{"migration_name": "test", "applied_at": "2023-01-01"}]
                mock_migrator.get_migration_status.return_value = mock_status
                mock_migrator_class.return_value = mock_migrator
                
                result = await get_migration_status()
                
                assert result == mock_status
                mock_migrator.get_migration_status.assert_called_once()