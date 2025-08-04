"""Sample documents and test data for testing."""

import os
import tempfile
from typing import Dict, Any
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Sample text content for testing
SAMPLE_TEXT_CONTENT = """
This is a sample document for testing the LLM Query Retrieval System.

The document contains multiple paragraphs with different types of information.
It includes technical details about machine learning, natural language processing,
and document retrieval systems.

Machine learning is a subset of artificial intelligence that focuses on algorithms
that can learn from and make predictions on data. Common applications include
image recognition, natural language processing, and recommendation systems.

Natural language processing (NLP) is a field of artificial intelligence that
gives computers the ability to understand, interpret and manipulate human language.
NLP draws from many disciplines, including computer science and computational linguistics.

Document retrieval systems are designed to find and return documents that are
relevant to a user's query. These systems often use techniques like TF-IDF,
BM25, or more recently, dense vector representations from neural networks.

The system should be able to answer questions about machine learning,
natural language processing, and document retrieval based on this content.
"""

SAMPLE_INSURANCE_CONTENT = """
INSURANCE POLICY DOCUMENT

Policy Number: INS-2024-001
Policyholder: John Smith
Coverage Period: January 1, 2024 - December 31, 2024

COVERAGE DETAILS:
- Auto Insurance: $500,000 liability coverage
- Comprehensive Coverage: $50,000 with $500 deductible
- Collision Coverage: $50,000 with $1,000 deductible
- Medical Payments: $10,000 per person
- Uninsured Motorist: $250,000 per person

EXCLUSIONS:
- Racing or speed contests
- Commercial use of vehicle
- Intentional damage
- War or nuclear hazard

CLAIMS PROCESS:
To file a claim, contact our 24/7 claims hotline at 1-800-CLAIMS.
Claims must be reported within 30 days of the incident.
A claims adjuster will be assigned within 48 hours.

PREMIUM INFORMATION:
Monthly Premium: $150
Annual Premium: $1,800
Payment due on the 15th of each month.
"""

SAMPLE_LEGAL_CONTENT = """
EMPLOYMENT CONTRACT

This Employment Agreement is entered into between TechCorp Inc. ("Company") 
and Jane Doe ("Employee") effective January 1, 2024.

POSITION AND DUTIES:
Employee shall serve as Senior Software Engineer and shall perform duties
including but not limited to:
- Software development and maintenance
- Code review and testing
- Technical documentation
- Collaboration with cross-functional teams

COMPENSATION:
Base Salary: $120,000 annually
Bonus: Up to 20% of base salary based on performance
Stock Options: 1,000 shares vesting over 4 years

BENEFITS:
- Health insurance (company pays 80%)
- Dental and vision insurance
- 401(k) with 4% company match
- 20 days paid time off
- 10 sick days

TERMINATION:
Either party may terminate this agreement with 30 days written notice.
Company may terminate immediately for cause.
Employee entitled to severance equal to 2 weeks pay if terminated without cause.

CONFIDENTIALITY:
Employee agrees to maintain confidentiality of all proprietary information
and trade secrets during and after employment.
"""

class TestDocumentFactory:
    """Factory for creating test documents in various formats."""
    
    @staticmethod
    def create_sample_pdf(content: str = SAMPLE_TEXT_CONTENT) -> bytes:
        """Create a sample PDF document with the given content."""
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Split content into lines and add to PDF
        lines = content.split('\n')
        y_position = 750
        
        for line in lines:
            if y_position < 50:  # Start new page if needed
                p.showPage()
                y_position = 750
            
            if line.strip():  # Only add non-empty lines
                p.drawString(50, y_position, line.strip())
            y_position -= 20
        
        p.save()
        buffer.seek(0)
        return buffer.getvalue()
    
    @staticmethod
    def create_sample_docx(content: str = SAMPLE_TEXT_CONTENT) -> bytes:
        """Create a sample DOCX document with the given content."""
        doc = Document()
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                doc.add_paragraph(paragraph.strip())
        
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()
    
    @staticmethod
    def create_sample_email() -> str:
        """Create a sample email content."""
        return """From: sender@example.com
To: recipient@example.com
Subject: Test Email for Document Processing
Date: Mon, 1 Jan 2024 12:00:00 +0000

This is a test email for the document processing system.

The email contains important information about the project timeline
and deliverables that need to be completed by the end of the quarter.

Key points:
- Project deadline: March 31, 2024
- Budget allocation: $50,000
- Team size: 5 developers
- Technology stack: Python, FastAPI, PostgreSQL

Please review and provide feedback by Friday.

Best regards,
Project Manager
"""

# Sample test data for various scenarios
SAMPLE_QUESTIONS = [
    "What is machine learning?",
    "How does natural language processing work?",
    "What are document retrieval systems?",
    "What techniques are used in document retrieval?",
]

INSURANCE_QUESTIONS = [
    "What is the policy number?",
    "What is the liability coverage amount?",
    "What is the deductible for comprehensive coverage?",
    "How long do I have to report a claim?",
    "What is the monthly premium?",
]

LEGAL_QUESTIONS = [
    "What is the employee's position?",
    "What is the base salary?",
    "How many vacation days are provided?",
    "What is the notice period for termination?",
    "What stock options are included?",
]

# Expected answers for validation
EXPECTED_ANSWERS = {
    "What is machine learning?": "machine learning is a subset of artificial intelligence",
    "What is the policy number?": "INS-2024-001",
    "What is the employee's position?": "Senior Software Engineer",
}

# Test metadata
TEST_DOCUMENT_METADATA = {
    "sample_doc": {
        "url": "http://example.com/sample.pdf",
        "content_type": "application/pdf",
        "size": 1024,
    },
    "insurance_doc": {
        "url": "http://example.com/insurance.pdf", 
        "content_type": "application/pdf",
        "size": 2048,
    },
    "legal_doc": {
        "url": "http://example.com/contract.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "size": 1536,
    },
}

def get_sample_chunks():
    """Get sample document chunks for testing."""
    return [
        {
            "id": "chunk_1",
            "document_id": "doc_1",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data.",
            "metadata": {"page": 1, "position": 0},
        },
        {
            "id": "chunk_2", 
            "document_id": "doc_1",
            "content": "Natural language processing (NLP) is a field of artificial intelligence that gives computers the ability to understand, interpret and manipulate human language.",
            "metadata": {"page": 1, "position": 1},
        },
        {
            "id": "chunk_3",
            "document_id": "doc_1", 
            "content": "Document retrieval systems are designed to find and return documents that are relevant to a user's query.",
            "metadata": {"page": 2, "position": 0},
        },
    ]

def get_sample_embeddings():
    """Get sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500-dimensional embedding
        [0.2, 0.3, 0.4, 0.5, 0.6] * 100,
        [0.3, 0.4, 0.5, 0.6, 0.7] * 100,
    ]