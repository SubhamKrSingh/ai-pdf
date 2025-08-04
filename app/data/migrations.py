"""
Database schema migration scripts and table creation.
Implements requirements 9.4, 8.4
"""

import logging
from typing import List, Dict, Any
import asyncpg
from asyncpg import Connection

from app.config import get_settings

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Custom exception for migration operations."""
    
    def __init__(self, message: str, migration_name: str, details: Dict[str, Any] = None):
        self.message = message
        self.migration_name = migration_name
        self.details = details or {}
        super().__init__(self.message)


class DatabaseMigrator:
    """
    Database migration manager for schema creation and updates.
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    async def create_connection(self) -> Connection:
        """
        Create a direct database connection for migrations.
        
        Returns:
            Connection: Database connection
            
        Raises:
            MigrationError: If connection fails
        """
        try:
            conn = await asyncpg.connect(self.settings.database_url)
            return conn
        except Exception as e:
            logger.error(f"Failed to create migration connection: {str(e)}")
            raise MigrationError(
                message="Failed to create database connection",
                migration_name="connection",
                details={"error": str(e)}
            )
    
    async def create_migrations_table(self, conn: Connection) -> None:
        """
        Create migrations tracking table if it doesn't exist.
        
        Args:
            conn: Database connection
            
        Raises:
            MigrationError: If table creation fails
        """
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    migration_name VARCHAR(255) NOT NULL UNIQUE,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum VARCHAR(64)
                )
            """)
            logger.info("Created schema_migrations table")
        except Exception as e:
            logger.error(f"Failed to create migrations table: {str(e)}")
            raise MigrationError(
                message="Failed to create migrations table",
                migration_name="schema_migrations",
                details={"error": str(e)}
            )
    
    async def is_migration_applied(self, conn: Connection, migration_name: str) -> bool:
        """
        Check if a migration has already been applied.
        
        Args:
            conn: Database connection
            migration_name: Name of the migration
            
        Returns:
            bool: True if migration has been applied
        """
        try:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM schema_migrations WHERE migration_name = $1",
                migration_name
            )
            return result > 0
        except Exception:
            # If table doesn't exist, migration hasn't been applied
            return False
    
    async def record_migration(self, conn: Connection, migration_name: str, checksum: str = None) -> None:
        """
        Record that a migration has been applied.
        
        Args:
            conn: Database connection
            migration_name: Name of the migration
            checksum: Optional checksum of the migration
        """
        try:
            await conn.execute(
                """
                INSERT INTO schema_migrations (migration_name, checksum, applied_at)
                VALUES ($1, $2, CURRENT_TIMESTAMP)
                """,
                migration_name, checksum
            )
            logger.info(f"Recorded migration: {migration_name}")
        except Exception as e:
            logger.error(f"Failed to record migration {migration_name}: {str(e)}")
            raise MigrationError(
                message="Failed to record migration",
                migration_name=migration_name,
                details={"error": str(e)}
            )
    
    async def create_documents_table(self, conn: Connection) -> None:
        """
        Create documents table for document metadata storage.
        
        Args:
            conn: Database connection
            
        Raises:
            MigrationError: If table creation fails
        """
        migration_name = "001_create_documents_table"
        
        if await self.is_migration_applied(conn, migration_name):
            logger.info(f"Migration {migration_name} already applied")
            return
        
        try:
            await conn.execute("""
                CREATE TABLE documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    url TEXT NOT NULL,
                    content_type VARCHAR(50),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'processing',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for better query performance
                CREATE INDEX idx_documents_url ON documents(url);
                CREATE INDEX idx_documents_status ON documents(status);
                CREATE INDEX idx_documents_processed_at ON documents(processed_at);
                
                -- Create trigger to update updated_at timestamp
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
                
                CREATE TRIGGER update_documents_updated_at 
                    BEFORE UPDATE ON documents 
                    FOR EACH ROW 
                    EXECUTE FUNCTION update_updated_at_column();
            """)
            
            await self.record_migration(conn, migration_name)
            logger.info("Created documents table with indexes and triggers")
            
        except Exception as e:
            logger.error(f"Failed to create documents table: {str(e)}")
            raise MigrationError(
                message="Failed to create documents table",
                migration_name=migration_name,
                details={"error": str(e)}
            )
    
    async def create_query_sessions_table(self, conn: Connection) -> None:
        """
        Create query_sessions table for query logging and analytics.
        
        Args:
            conn: Database connection
            
        Raises:
            MigrationError: If table creation fails
        """
        migration_name = "002_create_query_sessions_table"
        
        if await self.is_migration_applied(conn, migration_name):
            logger.info(f"Migration {migration_name} already applied")
            return
        
        try:
            await conn.execute("""
                CREATE TABLE query_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    questions JSONB NOT NULL,
                    answers JSONB NOT NULL,
                    processing_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
                
                -- Create indexes for better query performance
                CREATE INDEX idx_query_sessions_document_id ON query_sessions(document_id);
                CREATE INDEX idx_query_sessions_created_at ON query_sessions(created_at);
                CREATE INDEX idx_query_sessions_processing_time ON query_sessions(processing_time_ms);
                
                -- Create GIN index for JSONB columns for efficient querying
                CREATE INDEX idx_query_sessions_questions_gin ON query_sessions USING GIN (questions);
                CREATE INDEX idx_query_sessions_answers_gin ON query_sessions USING GIN (answers);
                CREATE INDEX idx_query_sessions_metadata_gin ON query_sessions USING GIN (metadata);
            """)
            
            await self.record_migration(conn, migration_name)
            logger.info("Created query_sessions table with indexes")
            
        except Exception as e:
            logger.error(f"Failed to create query_sessions table: {str(e)}")
            raise MigrationError(
                message="Failed to create query_sessions table",
                migration_name=migration_name,
                details={"error": str(e)}
            )
    
    async def create_document_url_cache_table(self, conn: Connection) -> None:
        """
        Create document_url_cache table for URL-based document caching.
        
        Args:
            conn: Database connection
            
        Raises:
            MigrationError: If table creation fails
        """
        migration_name = "003_create_document_url_cache_table"
        
        if await self.is_migration_applied(conn, migration_name):
            logger.info(f"Migration {migration_name} already applied")
            return
        
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_url_cache (
                    url_hash VARCHAR(64) PRIMARY KEY,
                    url TEXT NOT NULL,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                );

                -- Index for fast lookups
                CREATE INDEX IF NOT EXISTS idx_document_url_cache_document_id ON document_url_cache(document_id);
                CREATE INDEX IF NOT EXISTS idx_document_url_cache_created_at ON document_url_cache(created_at);

                -- Add comment for documentation
                COMMENT ON TABLE document_url_cache IS 'Caches processed documents by URL hash to avoid reprocessing the same documents';
                COMMENT ON COLUMN document_url_cache.url_hash IS 'SHA256 hash of the document URL for fast lookups';
                COMMENT ON COLUMN document_url_cache.url IS 'Original document URL';
                COMMENT ON COLUMN document_url_cache.document_id IS 'Reference to the processed document';
                COMMENT ON COLUMN document_url_cache.created_at IS 'When this cache entry was created';
            """)
            
            await self.record_migration(conn, migration_name)
            logger.info("Created document_url_cache table with indexes")
            
        except Exception as e:
            logger.error(f"Failed to create document_url_cache table: {str(e)}")
            raise MigrationError(
                message="Failed to create document_url_cache table",
                migration_name=migration_name,
                details={"error": str(e)}
            )
    
    async def create_extensions(self, conn: Connection) -> None:
        """
        Create required PostgreSQL extensions.
        
        Args:
            conn: Database connection
            
        Raises:
            MigrationError: If extension creation fails
        """
        migration_name = "000_create_extensions"
        
        if await self.is_migration_applied(conn, migration_name):
            logger.info(f"Migration {migration_name} already applied")
            return
        
        try:
            # Enable UUID generation extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            
            # Enable additional useful extensions
            await conn.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")
            
            await self.record_migration(conn, migration_name)
            logger.info("Created required PostgreSQL extensions")
            
        except Exception as e:
            logger.error(f"Failed to create extensions: {str(e)}")
            raise MigrationError(
                message="Failed to create extensions",
                migration_name=migration_name,
                details={"error": str(e)}
            )
    
    async def run_all_migrations(self) -> List[str]:
        """
        Run all pending migrations in order.
        
        Returns:
            List[str]: List of applied migration names
            
        Raises:
            MigrationError: If any migration fails
        """
        applied_migrations = []
        conn = None
        
        try:
            conn = await self.create_connection()
            
            # Create migrations table first
            await self.create_migrations_table(conn)
            
            # Run migrations in order
            migrations = [
                ("000_create_extensions", self.create_extensions),
                ("001_create_documents_table", self.create_documents_table),
                ("002_create_query_sessions_table", self.create_query_sessions_table),
                ("003_create_document_url_cache_table", self.create_document_url_cache_table),
            ]
            
            for migration_name, migration_func in migrations:
                try:
                    await migration_func(conn)
                    applied_migrations.append(migration_name)
                except MigrationError:
                    # Re-raise migration errors
                    raise
                except Exception as e:
                    # Wrap other exceptions
                    raise MigrationError(
                        message=f"Unexpected error in migration {migration_name}",
                        migration_name=migration_name,
                        details={"error": str(e)}
                    )
            
            logger.info(f"Successfully applied {len(applied_migrations)} migrations")
            return applied_migrations
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            if not isinstance(e, MigrationError):
                raise MigrationError(
                    message="Migration process failed",
                    migration_name="run_all_migrations",
                    details={"error": str(e)}
                )
            raise
        finally:
            if conn:
                await conn.close()
    
    async def get_migration_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all migrations.
        
        Returns:
            List[Dict[str, Any]]: Migration status information
        """
        conn = None
        try:
            conn = await self.create_connection()
            
            # Check if migrations table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'schema_migrations'
                )
            """)
            
            if not table_exists:
                return [{"status": "migrations_table_not_found"}]
            
            # Get applied migrations
            rows = await conn.fetch("""
                SELECT migration_name, applied_at, checksum
                FROM schema_migrations
                ORDER BY applied_at
            """)
            
            migrations = []
            for row in rows:
                migrations.append({
                    "migration_name": row["migration_name"],
                    "applied_at": row["applied_at"].isoformat(),
                    "checksum": row["checksum"]
                })
            
            return migrations
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {str(e)}")
            return [{"status": "error", "error": str(e)}]
        finally:
            if conn:
                await conn.close()


async def run_migrations() -> List[str]:
    """
    Convenience function to run all database migrations.
    
    Returns:
        List[str]: List of applied migration names
        
    Raises:
        MigrationError: If migrations fail
    """
    migrator = DatabaseMigrator()
    return await migrator.run_all_migrations()


async def get_migration_status() -> List[Dict[str, Any]]:
    """
    Convenience function to get migration status.
    
    Returns:
        List[Dict[str, Any]]: Migration status information
    """
    migrator = DatabaseMigrator()
    return await migrator.get_migration_status()