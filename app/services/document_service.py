"""
Document processing service that orchestrates download, parsing, chunking, and embedding storage.

This service implements the complete document processing pipeline with async processing,
error handling, progress tracking, and metadata management.

Requirements implemented: 2.1, 2.2, 2.3, 2.4, 3.1, 4.1, 4.2, 8.1, 8.2
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
from enum import Enum

from app.config import get_settings
from app.models.schemas import DocumentChunk, DocumentMetadata
from app.utils.document_downloader import DocumentDownloader, DocumentDownloadError
from app.utils.parsers.document_parser import DocumentParser, DocumentParseError, UnsupportedDocumentTypeError
from app.utils.text_chunker import TextChunker, ChunkingConfig
from app.services.embedding_service import get_embedding_service, EmbeddingServiceError
from app.services.document_cache_service import get_document_cache_service
from app.data.vector_store import get_vector_store, VectorStoreError
from app.data.repository import get_repository, DatabaseError


logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Enumeration of document processing stages."""
    INITIALIZING = "initializing"
    DOWNLOADING = "downloading"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStatus:
    """Class to track processing status and progress."""
    
    def __init__(self, document_id: str):
        self.document_id = document_id
        self.stage = ProcessingStage.INITIALIZING
        self.progress_percent = 0.0
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.callbacks: List[Callable] = []
    
    def update_stage(self, stage: ProcessingStage, progress: float = None):
        """Update processing stage and progress."""
        self.stage = stage
        self.stage_start_time = time.time()
        if progress is not None:
            self.progress_percent = progress
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def set_error(self, error_message: str):
        """Set error state."""
        self.stage = ProcessingStage.FAILED
        self.error_message = error_message
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")
    
    def add_callback(self, callback: Callable):
        """Add progress callback."""
        self.callbacks.append(callback)
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_stage_time(self) -> float:
        """Get current stage elapsed time in seconds."""
        return time.time() - self.stage_start_time


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    
    def __init__(self, message: str, stage: ProcessingStage, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.stage = stage
        self.details = details or {}
        super().__init__(self.message)


class DocumentProcessingService:
    """
    Service for orchestrating complete document processing pipeline.
    
    Handles download, parsing, chunking, embedding generation, and storage
    with comprehensive error handling and progress tracking.
    """
    
    def __init__(self):
        """Initialize the document processing service."""
        self.settings = get_settings()
        
        # Initialize components
        self.downloader = DocumentDownloader(
            timeout=self.settings.request_timeout,
            max_size_mb=self.settings.max_document_size_mb
        )
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            ChunkingConfig(
                chunk_size=self.settings.max_chunk_size,
                chunk_overlap=self.settings.chunk_overlap
            )
        )
        
        # Track active processing sessions
        self._active_sessions: Dict[str, ProcessingStatus] = {}
    
    async def process_document(
        self, 
        url: str, 
        progress_callback: Optional[Callable[[ProcessingStatus], None]] = None
    ) -> str:
        """
        Process a document through the complete pipeline with caching support.
        
        Args:
            url: URL of the document to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Document ID of the processed document
            
        Raises:
            DocumentProcessingError: If processing fails at any stage
        """
        # Check cache first for massive performance improvement
        cache_service = get_document_cache_service()
        cached_document_id = await cache_service.get_cached_document(url)
        
        if cached_document_id:
            logger.info(f"Using cached document {cached_document_id} for URL: {url}")
            
            # Create a minimal status for cached documents
            if progress_callback:
                status = ProcessingStatus(cached_document_id)
                status.add_callback(progress_callback)
                status.update_stage(ProcessingStage.COMPLETED, 100.0)
            
            return cached_document_id
        
        # No cache hit, proceed with full processing
        document_id = str(uuid4())
        status = ProcessingStatus(document_id)
        
        if progress_callback:
            status.add_callback(progress_callback)
        
        # Track this session
        self._active_sessions[document_id] = status
        
        try:
            logger.info(f"Starting document processing for {url} (ID: {document_id})")
            
            # Stage 1: Download document
            status.update_stage(ProcessingStage.DOWNLOADING, 10.0)
            content, content_type = await self._download_document(url, status)
            
            # Stage 2: Parse document
            status.update_stage(ProcessingStage.PARSING, 25.0)
            text_content = await self._parse_document(content, content_type, status)
            
            # Stage 3: Chunk document
            status.update_stage(ProcessingStage.CHUNKING, 40.0)
            chunks = await self._chunk_document(text_content, document_id, status)
            
            # Stage 4: Generate embeddings
            status.update_stage(ProcessingStage.EMBEDDING, 60.0)
            embedded_chunks = await self._generate_embeddings(chunks, status)
            
            # Stage 5: Store in vector database
            status.update_stage(ProcessingStage.STORING, 80.0)
            await self._store_vectors(embedded_chunks, status)
            
            # Stage 6: Store metadata
            await self._store_metadata(
                document_id, url, content_type, len(embedded_chunks), 
                len(content), status.get_elapsed_time() * 1000
            )
            
            # Stage 7: Cache the processed document
            await cache_service.cache_document(url, document_id)
            
            # Complete processing
            status.update_stage(ProcessingStage.COMPLETED, 100.0)
            logger.info(f"Document processing completed for {document_id} in {status.get_elapsed_time():.2f}s")
            
            return document_id
            
        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            logger.error(f"{error_msg} (Document ID: {document_id})")
            status.set_error(error_msg)
            
            # Store failed metadata
            try:
                await self._store_metadata(
                    document_id, url, "unknown", 0, 0, 
                    status.get_elapsed_time() * 1000, "failed", str(e)
                )
            except Exception as meta_error:
                logger.error(f"Failed to store error metadata: {meta_error}")
            
            if isinstance(e, DocumentProcessingError):
                raise
            else:
                raise DocumentProcessingError(
                    error_msg, 
                    status.stage, 
                    {"document_id": document_id, "url": url, "error": str(e)}
                )
        
        finally:
            # Clean up session tracking
            self._active_sessions.pop(document_id, None)
    
    async def _download_document(self, url: str, status: ProcessingStatus) -> tuple[bytes, str]:
        """Download document with error handling."""
        try:
            logger.debug(f"Downloading document from: {url}")
            content, content_type = await self.downloader.download_document(url)
            
            status.metadata.update({
                "url": url,
                "content_type": content_type,
                "file_size_bytes": len(content)
            })
            
            logger.info(f"Downloaded {len(content)} bytes, content-type: {content_type}")
            return content, content_type
            
        except DocumentDownloadError as e:
            raise DocumentProcessingError(
                f"Document download failed: {e}",
                ProcessingStage.DOWNLOADING,
                {"url": url, "error": str(e)}
            )
        except Exception as e:
            raise DocumentProcessingError(
                f"Unexpected download error: {e}",
                ProcessingStage.DOWNLOADING,
                {"url": url, "error": str(e)}
            )
    
    async def _parse_document(self, content: bytes, content_type: str, status: ProcessingStatus) -> str:
        """Parse document content with error handling."""
        try:
            logger.debug(f"Parsing document with content type: {content_type}")
            
            # Run parsing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text_content = await loop.run_in_executor(
                None, self.parser.parse_document, content, content_type
            )
            
            status.metadata.update({
                "text_length": len(text_content),
                "word_count": len(text_content.split())
            })
            
            logger.info(f"Parsed document: {len(text_content)} characters, {len(text_content.split())} words")
            return text_content
            
        except (DocumentParseError, UnsupportedDocumentTypeError) as e:
            raise DocumentProcessingError(
                f"Document parsing failed: {e}",
                ProcessingStage.PARSING,
                {"content_type": content_type, "error": str(e)}
            )
        except Exception as e:
            raise DocumentProcessingError(
                f"Unexpected parsing error: {e}",
                ProcessingStage.PARSING,
                {"content_type": content_type, "error": str(e)}
            )
    
    async def _chunk_document(self, text: str, document_id: str, status: ProcessingStatus) -> List[DocumentChunk]:
        """Chunk document text with error handling."""
        try:
            logger.debug(f"Chunking document text: {len(text)} characters")
            
            # Run chunking in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None, 
                self.chunker.chunk_text, 
                text, 
                document_id, 
                status.metadata.copy()
            )
            
            # Get chunking statistics
            chunk_stats = self.chunker.get_chunk_statistics(chunks)
            status.metadata.update({
                "chunk_count": len(chunks),
                "chunk_stats": chunk_stats
            })
            
            logger.info(f"Created {len(chunks)} chunks with average size {chunk_stats.get('avg_chunk_size', 0):.0f} characters")
            return chunks
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Document chunking failed: {e}",
                ProcessingStage.CHUNKING,
                {"text_length": len(text), "error": str(e)}
            )
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk], status: ProcessingStatus) -> List[DocumentChunk]:
        """Generate embeddings for chunks with error handling."""
        try:
            logger.debug(f"Generating embeddings for {len(chunks)} chunks")
            
            embedding_service = await get_embedding_service()
            
            # Process chunks in smaller batches with resilient error handling
            batch_size = 5  # Reduced batch size for better reliability
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            failed_batches = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch]
                batch_num = i // batch_size + 1
                
                try:
                    # Generate embeddings for batch with timeout handling
                    embeddings = await embedding_service.generate_embeddings(batch_texts)
                    
                    # Assign embeddings to chunks
                    for chunk, embedding in zip(batch, embeddings):
                        chunk.embedding = embedding
                    
                    logger.debug(f"Processed embedding batch {batch_num}/{total_batches}")
                    
                except EmbeddingServiceError as e:
                    logger.warning(f"Batch {batch_num} failed, will retry with smaller chunks: {e.message}")
                    failed_batches.append((i, batch))
                    
                    # Continue with other batches instead of failing completely
                    continue
                
                # Update progress
                progress = 60.0 + (20.0 * batch_num / total_batches)  # 60-80% range
                status.progress_percent = progress
            
            # Retry failed batches with individual chunk processing
            if failed_batches:
                logger.info(f"Retrying {len(failed_batches)} failed batches with individual processing")
                for batch_start, batch in failed_batches:
                    for j, chunk in enumerate(batch):
                        try:
                            embeddings = await embedding_service.generate_embeddings([chunk.content])
                            chunk.embedding = embeddings[0]
                            logger.debug(f"Recovered chunk {batch_start + j} individually")
                        except Exception as e:
                            logger.error(f"Failed to process chunk {batch_start + j} individually: {e}")
                            # Set a placeholder embedding or skip this chunk
                            chunk.embedding = None
            
            # Filter out chunks without embeddings
            valid_chunks = [chunk for chunk in chunks if chunk.embedding is not None]
            failed_chunks = len(chunks) - len(valid_chunks)
            
            if failed_chunks > 0:
                logger.warning(f"Failed to generate embeddings for {failed_chunks} chunks, continuing with {len(valid_chunks)} valid chunks")
            
            if not valid_chunks:
                raise DocumentProcessingError(
                    "No valid embeddings generated for any chunks",
                    ProcessingStage.EMBEDDING,
                    {"total_chunks": len(chunks), "failed_chunks": failed_chunks}
                )
            
            status.metadata.update({
                "embedding_dimension": len(valid_chunks[0].embedding) if valid_chunks else 0,
                "embedding_model": self.settings.jina_model,
                "successful_chunks": len(valid_chunks),
                "failed_chunks": failed_chunks
            })
            
            logger.info(f"Generated embeddings for {len(valid_chunks)} out of {len(chunks)} chunks")
            return valid_chunks
            
        except EmbeddingServiceError as e:
            raise DocumentProcessingError(
                f"Embedding generation failed: {e.message}",
                ProcessingStage.EMBEDDING,
                {"chunk_count": len(chunks), "error_code": e.error_code, "details": e.details}
            )
        except Exception as e:
            raise DocumentProcessingError(
                f"Unexpected embedding error: {e}",
                ProcessingStage.EMBEDDING,
                {"chunk_count": len(chunks), "error": str(e)}
            )
    
    async def _store_vectors(self, chunks: List[DocumentChunk], status: ProcessingStatus) -> None:
        """Store vectors in Pinecone with error handling."""
        try:
            logger.debug(f"Storing {len(chunks)} vectors in Pinecone")
            
            vector_store = await get_vector_store()
            success = await vector_store.store_vectors(chunks)
            
            if not success:
                raise DocumentProcessingError(
                    "Vector storage returned failure status",
                    ProcessingStage.STORING,
                    {"chunk_count": len(chunks)}
                )
            
            status.metadata.update({
                "vectors_stored": len(chunks),
                "vector_store": "pinecone"
            })
            
            logger.info(f"Successfully stored {len(chunks)} vectors")
            
        except VectorStoreError as e:
            raise DocumentProcessingError(
                f"Vector storage failed: {e.message}",
                ProcessingStage.STORING,
                {"chunk_count": len(chunks), "error_code": e.error_code, "details": e.details}
            )
        except Exception as e:
            raise DocumentProcessingError(
                f"Unexpected vector storage error: {e}",
                ProcessingStage.STORING,
                {"chunk_count": len(chunks), "error": str(e)}
            )
    
    async def _store_metadata(
        self, 
        document_id: str, 
        url: str, 
        content_type: str, 
        chunk_count: int,
        file_size: int,
        processing_time_ms: float,
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> None:
        """Store document metadata in PostgreSQL."""
        try:
            repository = await get_repository()
            await repository.store_document_metadata(
                document_id=document_id,
                url=url,
                content_type=content_type,
                chunk_count=chunk_count,
                status=status
            )
            
            logger.debug(f"Stored metadata for document {document_id}")
            
        except DatabaseError as e:
            logger.error(f"Failed to store document metadata: {e.message}")
            # Don't raise here as this is not critical for the main processing
        except Exception as e:
            logger.error(f"Unexpected metadata storage error: {e}")
    
    def get_processing_status(self, document_id: str) -> Optional[ProcessingStatus]:
        """Get current processing status for a document."""
        return self._active_sessions.get(document_id)
    
    def get_active_sessions(self) -> Dict[str, ProcessingStatus]:
        """Get all active processing sessions."""
        return self._active_sessions.copy()
    
    async def cancel_processing(self, document_id: str) -> bool:
        """
        Cancel document processing (if possible).
        
        Args:
            document_id: ID of document to cancel
            
        Returns:
            bool: True if cancellation was successful
        """
        status = self._active_sessions.get(document_id)
        if not status:
            return False
        
        # Set error state to indicate cancellation
        status.set_error("Processing cancelled by user")
        
        # Clean up session
        self._active_sessions.pop(document_id, None)
        
        logger.info(f"Cancelled processing for document {document_id}")
        return True
    
    async def reprocess_document(self, document_id: str, url: str) -> str:
        """
        Reprocess an existing document.
        
        Args:
            document_id: Existing document ID
            url: Document URL
            
        Returns:
            str: Document ID (same as input)
        """
        # First, clean up existing data
        try:
            vector_store = await get_vector_store()
            await vector_store.delete_document_vectors(document_id)
            logger.info(f"Cleaned up existing vectors for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up existing vectors: {e}")
        
        # Process with existing document ID
        status = ProcessingStatus(document_id)
        self._active_sessions[document_id] = status
        
        try:
            # Follow same processing pipeline
            status.update_stage(ProcessingStage.DOWNLOADING, 10.0)
            content, content_type = await self._download_document(url, status)
            
            status.update_stage(ProcessingStage.PARSING, 25.0)
            text_content = await self._parse_document(content, content_type, status)
            
            status.update_stage(ProcessingStage.CHUNKING, 40.0)
            chunks = await self._chunk_document(text_content, document_id, status)
            
            status.update_stage(ProcessingStage.EMBEDDING, 60.0)
            embedded_chunks = await self._generate_embeddings(chunks, status)
            
            status.update_stage(ProcessingStage.STORING, 80.0)
            await self._store_vectors(embedded_chunks, status)
            
            await self._store_metadata(
                document_id, url, content_type, len(embedded_chunks),
                len(content), status.get_elapsed_time() * 1000
            )
            
            status.update_stage(ProcessingStage.COMPLETED, 100.0)
            logger.info(f"Document reprocessing completed for {document_id}")
            
            return document_id
            
        except Exception as e:
            error_msg = f"Document reprocessing failed: {str(e)}"
            status.set_error(error_msg)
            raise DocumentProcessingError(
                error_msg, 
                status.stage, 
                {"document_id": document_id, "url": url, "error": str(e)}
            )
        finally:
            self._active_sessions.pop(document_id, None)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health_status = {
            "service": "document_processing",
            "status": "healthy",
            "timestamp": time.time(),
            "active_sessions": len(self._active_sessions),
            "components": {}
        }
        
        # Check embedding service
        try:
            embedding_service = await get_embedding_service()
            cache_stats = embedding_service.get_cache_stats()
            health_status["components"]["embedding_service"] = {
                "status": "healthy",
                "cache_stats": cache_stats
            }
        except Exception as e:
            health_status["components"]["embedding_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check vector store
        try:
            vector_store = await get_vector_store()
            await vector_store.health_check()
            index_stats = await vector_store.get_index_stats()
            health_status["components"]["vector_store"] = {
                "status": "healthy",
                "index_stats": index_stats
            }
        except Exception as e:
            health_status["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check repository
        try:
            repository = await get_repository()
            db_health = await repository.health_check()
            health_status["components"]["repository"] = {
                "status": "healthy",
                "database": db_health
            }
        except Exception as e:
            health_status["components"]["repository"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status


# Global service instance
_document_service: Optional[DocumentProcessingService] = None


def get_document_service() -> DocumentProcessingService:
    """
    Get or create the global document processing service instance.
    
    Returns:
        DocumentProcessingService: The service instance
    """
    global _document_service
    if _document_service is None:
        _document_service = DocumentProcessingService()
    return _document_service


async def process_document_from_url(
    url: str, 
    progress_callback: Optional[Callable[[ProcessingStatus], None]] = None
) -> str:
    """
    Convenience function to process a document from URL.
    
    Args:
        url: Document URL to process
        progress_callback: Optional progress callback
        
    Returns:
        str: Document ID
    """
    service = get_document_service()
    return await service.process_document(url, progress_callback)