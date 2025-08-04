"""
PostgreSQL database repository for document metadata and query session management.
Implements requirements 9.4, 8.4
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
import asyncpg
from asyncpg import Pool, Connection
from contextlib import asynccontextmanager

from app.config import get_settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations."""
    
    def __init__(self, message: str, operation: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.operation = operation
        self.details = details or {}
        super().__init__(self.message)


class DatabaseRepository:
    """
    PostgreSQL repository for document metadata and query session management.
    Provides connection pooling and comprehensive error handling.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._pool: Optional[Pool] = None
        self._connection_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """
        Initialize database connection pool.
        
        Raises:
            DatabaseError: If connection pool creation fails
        """
        try:
            async with self._connection_lock:
                if self._pool is None:
                    logger.info("Initializing database connection pool")
                    self._pool = await asyncpg.create_pool(
                        self.settings.database_url,
                        min_size=5,
                        max_size=20,
                        command_timeout=30,
                        server_settings={
                            'application_name': 'llm_query_retrieval_system',
                            'timezone': 'UTC'
                        }
                    )
                    logger.info("Database connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {str(e)}")
            raise DatabaseError(
                message="Failed to initialize database connection pool",
                operation="initialize",
                details={"error": str(e)}
            )
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            logger.info("Closing database connection pool")
            await self._pool.close()
            self._pool = None
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get database connection from pool with proper error handling.
        
        Yields:
            Connection: Database connection
            
        Raises:
            DatabaseError: If connection cannot be acquired
        """
        if not self._pool:
            await self.initialize()
        
        connection = None
        try:
            connection = await self._pool.acquire()
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise DatabaseError(
                message="Database connection error",
                operation="get_connection",
                details={"error": str(e)}
            )
        finally:
            if connection:
                await self._pool.release(connection)
    
    async def store_document_metadata(
        self, 
        document_id: str, 
        url: str, 
        content_type: str, 
        chunk_count: int,
        status: str = "completed"
    ) -> bool:
        """
        Store document metadata in the database.
        
        Args:
            document_id: Unique document identifier
            url: Document source URL
            content_type: MIME type of the document
            chunk_count: Number of chunks created from document
            status: Processing status (default: "completed")
            
        Returns:
            bool: True if storage was successful
            
        Raises:
            DatabaseError: If storage operation fails
        """
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO documents (id, url, content_type, chunk_count, status, processed_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (id) DO UPDATE SET
                        url = EXCLUDED.url,
                        content_type = EXCLUDED.content_type,
                        chunk_count = EXCLUDED.chunk_count,
                        status = EXCLUDED.status,
                        processed_at = EXCLUDED.processed_at
                    """,
                    document_id, url, content_type, chunk_count, status, datetime.utcnow()
                )
                
                logger.info(f"Stored metadata for document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store document metadata: {str(e)}")
            raise DatabaseError(
                message="Failed to store document metadata",
                operation="store_document_metadata",
                details={
                    "document_id": document_id,
                    "url": url,
                    "error": str(e)
                }
            )
    
    async def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document metadata from the database.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Optional[Dict[str, Any]]: Document metadata or None if not found
            
        Raises:
            DatabaseError: If retrieval operation fails
        """
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, url, content_type, processed_at, chunk_count, status
                    FROM documents 
                    WHERE id = $1
                    """,
                    document_id
                )
                
                if row:
                    metadata = {
                        "id": str(row["id"]),
                        "url": row["url"],
                        "content_type": row["content_type"],
                        "processed_at": row["processed_at"].isoformat(),
                        "chunk_count": row["chunk_count"],
                        "status": row["status"]
                    }
                    logger.debug(f"Retrieved metadata for document {document_id}")
                    return metadata
                else:
                    logger.debug(f"No metadata found for document {document_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve document metadata: {str(e)}")
            raise DatabaseError(
                message="Failed to retrieve document metadata",
                operation="get_document_metadata",
                details={
                    "document_id": document_id,
                    "error": str(e)
                }
            )
    
    async def log_query_session(
        self,
        document_id: str,
        questions: List[str],
        answers: List[str],
        processing_time_ms: int
    ) -> str:
        """
        Log query session for analytics and debugging.
        
        Args:
            document_id: Document that was queried
            questions: List of questions asked
            answers: List of answers generated
            processing_time_ms: Total processing time in milliseconds
            
        Returns:
            str: Session ID
            
        Raises:
            DatabaseError: If logging operation fails
        """
        try:
            session_id = str(uuid4())
            
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO query_sessions (id, document_id, questions, answers, processing_time_ms, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    session_id,
                    document_id,
                    questions,  # PostgreSQL JSONB will handle the list serialization
                    answers,
                    processing_time_ms,
                    datetime.utcnow()
                )
                
                logger.info(f"Logged query session {session_id} for document {document_id}")
                return session_id
                
        except Exception as e:
            logger.error(f"Failed to log query session: {str(e)}")
            raise DatabaseError(
                message="Failed to log query session",
                operation="log_query_session",
                details={
                    "document_id": document_id,
                    "questions_count": len(questions),
                    "error": str(e)
                }
            )
    
    async def get_query_sessions(
        self, 
        document_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve query sessions for analytics.
        
        Args:
            document_id: Optional document ID to filter by
            limit: Maximum number of sessions to return
            
        Returns:
            List[Dict[str, Any]]: List of query sessions
            
        Raises:
            DatabaseError: If retrieval operation fails
        """
        try:
            async with self.get_connection() as conn:
                if document_id:
                    rows = await conn.fetch(
                        """
                        SELECT id, document_id, questions, answers, processing_time_ms, created_at
                        FROM query_sessions 
                        WHERE document_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                        """,
                        document_id, limit
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, document_id, questions, answers, processing_time_ms, created_at
                        FROM query_sessions 
                        ORDER BY created_at DESC
                        LIMIT $1
                        """,
                        limit
                    )
                
                sessions = []
                for row in rows:
                    session = {
                        "id": str(row["id"]),
                        "document_id": str(row["document_id"]),
                        "questions": row["questions"],
                        "answers": row["answers"],
                        "processing_time_ms": row["processing_time_ms"],
                        "created_at": row["created_at"].isoformat()
                    }
                    sessions.append(session)
                
                logger.debug(f"Retrieved {len(sessions)} query sessions")
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to retrieve query sessions: {str(e)}")
            raise DatabaseError(
                message="Failed to retrieve query sessions",
                operation="get_query_sessions",
                details={
                    "document_id": document_id,
                    "error": str(e)
                }
            )
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document and all associated query sessions.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            bool: True if deletion was successful
            
        Raises:
            DatabaseError: If deletion operation fails
        """
        try:
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Delete query sessions first (foreign key constraint)
                    await conn.execute(
                        "DELETE FROM query_sessions WHERE document_id = $1",
                        document_id
                    )
                    
                    # Delete document metadata
                    result = await conn.execute(
                        "DELETE FROM documents WHERE id = $1",
                        document_id
                    )
                    
                    # Check if document was actually deleted
                    deleted_count = int(result.split()[-1])
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted document {document_id} and associated sessions")
                        return True
                    else:
                        logger.warning(f"Document {document_id} not found for deletion")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise DatabaseError(
                message="Failed to delete document",
                operation="delete_document",
                details={
                    "document_id": document_id,
                    "error": str(e)
                }
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.
        
        Returns:
            Dict[str, Any]: Health check results
            
        Raises:
            DatabaseError: If health check fails
        """
        try:
            async with self.get_connection() as conn:
                # Test basic connectivity
                result = await conn.fetchval("SELECT 1")
                
                # Get connection pool stats
                pool_stats = {
                    "size": self._pool.get_size(),
                    "min_size": self._pool.get_min_size(),
                    "max_size": self._pool.get_max_size(),
                    "idle_size": self._pool.get_idle_size()
                }
                
                return {
                    "status": "healthy",
                    "connection_test": result == 1,
                    "pool_stats": pool_stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            raise DatabaseError(
                message="Database health check failed",
                operation="health_check",
                details={"error": str(e)}
            )


# Global repository instance
_repository: Optional[DatabaseRepository] = None


async def get_repository() -> DatabaseRepository:
    """
    Get global database repository instance.
    
    Returns:
        DatabaseRepository: Initialized repository instance
    """
    global _repository
    if _repository is None:
        _repository = DatabaseRepository()
        await _repository.initialize()
    return _repository


async def close_repository() -> None:
    """Close global database repository."""
    global _repository
    if _repository:
        await _repository.close()
        _repository = None