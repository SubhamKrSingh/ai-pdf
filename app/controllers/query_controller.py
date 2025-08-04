"""
Query controller that orchestrates document and query services.

This module implements the main controller logic that coordinates
document processing and query answering according to the system requirements.
"""

import logging
from typing import List
from fastapi import HTTPException

from app.models.schemas import QueryRequest, QueryResponse, DocumentMetadata
from app.services.document_service import get_document_service
from app.services.query_service import get_query_service
from app.config import get_settings

logger = logging.getLogger(__name__)


class QueryController:
    """
    Controller that orchestrates document processing and query answering.
    
    This class coordinates the document service and query service to provide
    the main functionality of the system according to requirements 1.1, 1.2, 1.4.
    """
    
    def __init__(self):
        """Initialize the controller with required services."""
        self.settings = get_settings()
        self.document_service = None
        self.query_service = None
    
    async def process_query_request(self, request: QueryRequest) -> QueryResponse:
        """
        Process a complete query request including document processing and question answering.
        
        This method implements the main workflow:
        1. Download and process the document
        2. Generate embeddings and store in vector database
        3. Process each question and generate answers
        4. Return structured response
        
        Args:
            request: QueryRequest containing document URL and questions
            
        Returns:
            QueryResponse: Structured response with answers
            
        Raises:
            HTTPException: For various error conditions during processing
        """
        document_url = str(request.documents)
        questions = request.questions
        
        logger.info(f"Starting query processing for document: {document_url}")
        logger.info(f"Questions to process: {len(questions)}")
        
        try:
            # Step 1: Process the document
            logger.info("Step 1: Processing document...")
            document_id = await self._process_document(document_url)
            logger.info(f"Document processed successfully with ID: {document_id}")
            
            # Step 2: Process questions and generate answers
            logger.info("Step 2: Processing questions...")
            answers = await self._process_questions(questions, document_id)
            logger.info(f"Generated {len(answers)} answers")
            
            # Step 3: Validate and return response
            if len(answers) != len(questions):
                raise ValueError(f"Answer count ({len(answers)}) does not match question count ({len(questions)})")
            
            response = QueryResponse(answers=answers)
            logger.info("Query processing completed successfully")
            
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {str(e)}"
            )
    
    async def _process_document(self, document_url: str) -> str:
        """
        Process a document through the complete pipeline.
        
        Args:
            document_url: URL of the document to process
            
        Returns:
            str: Document ID for the processed document
            
        Raises:
            HTTPException: If document processing fails
        """
        try:
            # Get document service instance
            if self.document_service is None:
                self.document_service = get_document_service()
            
            # Process document using the service's main method
            logger.debug("Processing document through complete pipeline...")
            document_id = await self.document_service.process_document(document_url)
            
            return document_id
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            if "download" in str(e).lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download document: {str(e)}"
                )
            elif "parse" in str(e).lower() or "unsupported" in str(e).lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to parse document: {str(e)}"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Document processing error: {str(e)}"
                )
    
    async def _process_questions(self, questions: List[str], document_id: str) -> List[str]:
        """
        Process a list of questions and generate answers.
        
        Args:
            questions: List of natural language questions
            document_id: ID of the processed document
            
        Returns:
            List[str]: List of answers corresponding to the questions
            
        Raises:
            HTTPException: If question processing fails
        """
        try:
            # Get query service instance
            if self.query_service is None:
                self.query_service = await get_query_service()
            
            # Process multiple questions using the service's method
            answers = await self.query_service.process_multiple_questions(questions, document_id)
            
            return answers
            
        except Exception as e:
            logger.error(f"Question processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Question processing error: {str(e)}"
            )
    
    async def health_check(self) -> dict:
        """
        Perform a health check of all dependent services.
        
        Returns:
            dict: Health status of the controller and its services
        """
        try:
            # Get service instances
            if self.document_service is None:
                self.document_service = get_document_service()
            if self.query_service is None:
                self.query_service = await get_query_service()
            
            # Check document service
            doc_health = await self.document_service.health_check()
            
            # Check query service  
            query_health = await self.query_service.health_check()
            
            return {
                "status": "healthy",
                "services": {
                    "document_service": doc_health,
                    "query_service": query_health
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }