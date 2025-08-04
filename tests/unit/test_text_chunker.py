"""
Unit tests for text chunking utilities.

Tests the recursive character text splitter with configurable chunk size and overlap,
semantic coherence preservation logic, and chunk metadata generation.

Tests requirements 3.1, 3.2, 3.3, 3.4.
"""

import pytest
from unittest.mock import patch
from typing import List, Dict, Any

from app.utils.text_chunker import (
    TextChunker,
    ChunkingConfig,
    create_chunker,
    chunk_document_text
)
from app.models.schemas import DocumentChunk


class TestChunkingConfig:
    """Test ChunkingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.keep_separator is True
        assert len(config.separators) > 0
        assert "\n\n" in config.separators
        assert ". " in config.separators
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n", " "],
            keep_separator=False
        )
        
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.separators == ["\n", " "]
        assert config.keep_separator is False


class TestTextChunker:
    """Test TextChunker class functionality."""
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        chunker = TextChunker()
        
        assert chunker.config.chunk_size == 1000
        assert chunker.config.chunk_overlap == 200
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        chunker = TextChunker(config)
        
        assert chunker.config.chunk_size == 500
        assert chunker.config.chunk_overlap == 50
    
    def test_init_invalid_overlap(self):
        """Test initialization fails with invalid overlap."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=150)
        
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            TextChunker(config)
    
    def test_init_invalid_chunk_size(self):
        """Test initialization fails with too small chunk size."""
        config = ChunkingConfig(chunk_size=30, chunk_overlap=10)
        
        with pytest.raises(ValueError, match="Chunk size must be at least 50 characters"):
            TextChunker(config)
    
    def test_chunk_text_empty_input(self):
        """Test chunking fails with empty text."""
        chunker = TextChunker()
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            chunker.chunk_text("", "doc_id")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            chunker.chunk_text("   ", "doc_id")
    
    def test_chunk_text_empty_document_id(self):
        """Test chunking fails with empty document ID."""
        chunker = TextChunker()
        
        with pytest.raises(ValueError, match="Document ID cannot be empty"):
            chunker.chunk_text("Some text", "")
        
        with pytest.raises(ValueError, match="Document ID cannot be empty"):
            chunker.chunk_text("Some text", "   ")
    
    def test_chunk_text_small_text(self):
        """Test chunking small text that fits in one chunk."""
        chunker = TextChunker()
        text = "This is a small text that should fit in one chunk."
        document_id = "test_doc_1"
        
        chunks = chunker.chunk_text(text, document_id)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].document_id == document_id
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)
    
    def test_chunk_text_paragraph_splitting(self):
        """Test chunking respects paragraph boundaries."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = TextChunker(config)
        
        text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph with additional content."
        document_id = "test_doc_2"
        
        chunks = chunker.chunk_text(text, document_id)
        
        assert len(chunks) > 1
        # Check that chunks maintain paragraph structure
        for chunk in chunks:
            assert len(chunk.content) <= config.chunk_size + 50  # Allow some flexibility
            assert chunk.document_id == document_id
    
    def test_chunk_text_sentence_splitting(self):
        """Test chunking respects sentence boundaries."""
        config = ChunkingConfig(chunk_size=80, chunk_overlap=15)
        chunker = TextChunker(config)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        document_id = "test_doc_3"
        
        chunks = chunker.chunk_text(text, document_id)
        
        assert len(chunks) > 1
        # Check that most chunks end with sentence boundaries
        sentence_endings = 0
        for chunk in chunks[:-1]:  # Exclude last chunk
            if chunk.content.rstrip().endswith(('.', '!', '?')):
                sentence_endings += 1
        
        # At least half should end with sentence boundaries
        assert sentence_endings >= len(chunks) // 2
    
    def test_chunk_text_word_splitting(self):
        """Test chunking falls back to word boundaries."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = TextChunker(config)
        
        # Text without sentence boundaries
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15"
        document_id = "test_doc_4"
        
        chunks = chunker.chunk_text(text, document_id)
        
        assert len(chunks) > 1
        # Check that chunks don't break words unnecessarily
        for chunk in chunks:
            words = chunk.content.split()
            # Each chunk should have complete words (except possibly at boundaries)
            assert len(words) > 0
    
    def test_chunk_text_overlap_functionality(self):
        """Test that chunks have proper overlap."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=30)
        chunker = TextChunker(config)
        
        text = "This is a long text that needs to be split into multiple chunks with overlap. " * 5
        document_id = "test_doc_5"
        
        chunks = chunker.chunk_text(text, document_id)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                current_end = chunks[i].content[-20:]  # Last 20 chars
                next_start = chunks[i + 1].content[:20]  # First 20 chars
                
                # There should be some common text (allowing for word boundaries)
                has_overlap = any(
                    word in next_start for word in current_end.split()[-3:]
                    if len(word) > 3
                )
                assert has_overlap or len(chunks[i].content) < config.chunk_size
    
    def test_chunk_metadata_generation(self):
        """Test comprehensive metadata generation."""
        chunker = TextChunker()
        text = "This is a test document. It has multiple sentences! Does it work properly?"
        document_id = "test_doc_6"
        base_metadata = {"source": "test", "type": "sample"}
        
        chunks = chunker.chunk_text(text, document_id, base_metadata)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Check basic metadata
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["total_chunks"] == 1
        assert chunk.metadata["chunk_size"] == len(text)
        assert chunk.metadata["word_count"] > 0
        assert chunk.metadata["sentence_count"] >= 2  # Should detect sentences
        
        # Check position metadata
        assert chunk.metadata["is_first_chunk"] is True
        assert chunk.metadata["is_last_chunk"] is True
        assert chunk.metadata["relative_position"] == 0.0
        
        # Check base metadata is preserved
        assert chunk.metadata["source"] == "test"
        assert chunk.metadata["type"] == "sample"
    
    def test_semantic_analysis(self):
        """Test semantic analysis of chunk content."""
        chunker = TextChunker()
        
        # Test technical content
        tech_text = "The API returns JSON data with HTTP status codes. Use SQL queries for database access."
        chunks = chunker.chunk_text(tech_text, "tech_doc")
        assert chunks[0].metadata["appears_technical"] is True
        
        # Test legal content
        legal_text = "The party shall hereby agree to the terms, whereas the contract is pursuant to law."
        chunks = chunker.chunk_text(legal_text, "legal_doc")
        assert chunks[0].metadata["appears_legal"] is True
        
        # Test financial content
        financial_text = "The total cost is $1,234.56 and the budget is €2,000.00 for this quarter."
        chunks = chunker.chunk_text(financial_text, "financial_doc")
        assert chunks[0].metadata["appears_financial"] is True
    
    def test_preprocess_text(self):
        """Test text preprocessing functionality."""
        chunker = TextChunker()
        
        # Test whitespace normalization
        messy_text = "Text   with    excessive     whitespace\n\n\n\nand\t\ttabs"
        processed = chunker._preprocess_text(messy_text)
        
        assert "   " not in processed  # No triple spaces
        assert "\n\n\n" not in processed  # No triple newlines
        assert "\t\t" not in processed  # No double tabs
    
    def test_find_chunk_position(self):
        """Test finding chunk positions in original text."""
        chunker = TextChunker()
        original_text = "This is the original text with multiple sentences. It should be searchable."
        chunk_content = "original text with multiple"
        
        start, end = chunker._find_chunk_position(original_text, chunk_content, 0)
        
        assert start >= 0
        assert end > start
        assert original_text[start:end] == chunk_content
    
    def test_get_chunk_statistics(self):
        """Test chunk statistics generation."""
        chunker = TextChunker()
        text = "This is a test document. " * 50  # Create longer text
        chunks = chunker.chunk_text(text, "stats_doc")
        
        stats = chunker.get_chunk_statistics(chunks)
        
        assert "total_chunks" in stats
        assert "avg_chunk_size" in stats
        assert "min_chunk_size" in stats
        assert "max_chunk_size" in stats
        assert "total_characters" in stats
        assert "total_words" in stats
        assert "overlap_efficiency" in stats
        
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_characters"] > 0
        assert 0 <= stats["overlap_efficiency"] <= 1
    
    def test_get_chunk_statistics_empty(self):
        """Test chunk statistics with empty chunk list."""
        chunker = TextChunker()
        stats = chunker.get_chunk_statistics([])
        
        assert stats["total_chunks"] == 0


class TestDocumentTypeHandling:
    """Test chunking with different document types."""
    
    def test_pdf_like_content(self):
        """Test chunking PDF-like content with page breaks."""
        chunker = TextChunker()
        
        pdf_text = """Page 1 Content
        
This is the first page of the document with some content.

Page 2 Content

This is the second page with different content and formatting.

Page 3 Content

Final page with conclusion and summary information."""
        
        chunks = chunker.chunk_text(pdf_text, "pdf_doc", {"content_type": "pdf"})
        
        assert len(chunks) >= 1
        # Check that page structure is somewhat preserved
        page_references = sum(1 for chunk in chunks if "Page" in chunk.content)
        assert page_references > 0
    
    def test_email_like_content(self):
        """Test chunking email-like content with headers."""
        chunker = TextChunker()
        
        email_text = """From: sender@example.com
To: recipient@example.com
Subject: Important Document

Dear Recipient,

This is the body of the email with important information that needs to be processed.

The email contains multiple paragraphs and should be chunked appropriately.

Best regards,
Sender"""
        
        chunks = chunker.chunk_text(email_text, "email_doc", {"content_type": "email"})
        
        assert len(chunks) >= 1
        # Check that email structure is preserved
        has_header = any("From:" in chunk.content or "Subject:" in chunk.content for chunk in chunks)
        assert has_header
    
    def test_docx_like_content(self):
        """Test chunking DOCX-like content with formatting."""
        chunker = TextChunker()
        
        docx_text = """DOCUMENT TITLE

Introduction

This document contains structured content with headers and sections.

Section 1: Overview

Content for the first section with detailed information.

Section 2: Details

More detailed content in the second section.

Conclusion

Final thoughts and summary of the document."""
        
        chunks = chunker.chunk_text(docx_text, "docx_doc", {"content_type": "docx"})
        
        assert len(chunks) >= 1
        # Check that document structure is preserved
        has_headers = any(chunk.metadata.get("has_headers", False) for chunk in chunks)
        assert has_headers


class TestFactoryFunctions:
    """Test factory and convenience functions."""
    
    def test_create_chunker(self):
        """Test chunker factory function."""
        chunker = create_chunker(chunk_size=500, chunk_overlap=100)
        
        assert chunker.config.chunk_size == 500
        assert chunker.config.chunk_overlap == 100
    
    def test_create_chunker_defaults(self):
        """Test chunker factory with default values."""
        chunker = create_chunker()
        
        assert chunker.config.chunk_size == 1000
        assert chunker.config.chunk_overlap == 200
    
    def test_chunk_document_text(self):
        """Test convenience function for document chunking."""
        text = "This is a test document for the convenience function."
        document_id = "convenience_test"
        
        chunks = chunk_document_text(text, document_id)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].document_id == document_id
    
    def test_chunk_document_text_custom_params(self):
        """Test convenience function with custom parameters."""
        text = "This is a longer test document. " * 20
        document_id = "custom_test"
        metadata = {"test": "value"}
        
        chunks = chunk_document_text(
            text, 
            document_id, 
            chunk_size=100, 
            chunk_overlap=20,
            metadata=metadata
        )
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.document_id == document_id
            assert chunk.metadata["test"] == "value"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_words(self):
        """Test handling of very long words that exceed chunk size."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = TextChunker(config)
        
        # Create text with a very long word
        long_word = "a" * 100
        text = f"Short words and {long_word} and more short words."
        
        chunks = chunker.chunk_text(text, "long_word_doc")
        
        assert len(chunks) >= 1
        # The long word should be handled somehow (likely split)
        total_content = "".join(chunk.content for chunk in chunks)
        assert long_word in total_content.replace(" ", "")
    
    def test_only_whitespace_separators(self):
        """Test text with only whitespace separators."""
        chunker = TextChunker()
        
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunks = chunker.chunk_text(text, "whitespace_doc")
        
        assert len(chunks) >= 1
        # All chunks should contain complete words
        for chunk in chunks:
            assert not chunk.content.startswith(" ")
            assert not chunk.content.endswith(" ")
    
    def test_unicode_content(self):
        """Test handling of Unicode content."""
        chunker = TextChunker()
        
        unicode_text = "This text contains émojis 🚀 and spëcial chäractërs. It should be handled properly."
        chunks = chunker.chunk_text(unicode_text, "unicode_doc")
        
        assert len(chunks) >= 1
        assert "🚀" in chunks[0].content
        assert "émojis" in chunks[0].content
    
    def test_mixed_line_endings(self):
        """Test handling of mixed line endings."""
        chunker = TextChunker()
        
        mixed_text = "Line 1\nLine 2\r\nLine 3\rLine 4"
        chunks = chunker.chunk_text(mixed_text, "mixed_endings_doc")
        
        assert len(chunks) >= 1
        # Should handle different line endings gracefully
        assert "Line 1" in chunks[0].content
        assert "Line 4" in chunks[0].content
    
    def test_empty_chunks_filtered(self):
        """Test that empty chunks are filtered out."""
        config = ChunkingConfig(chunk_size=60, chunk_overlap=10)
        chunker = TextChunker(config)
        
        # Text with lots of whitespace that might create empty chunks
        text = "Word1.    \n\n\n    Word2.    \n\n\n    Word3."
        chunks = chunker.chunk_text(text, "empty_chunks_doc")
        
        # All chunks should have content
        for chunk in chunks:
            assert chunk.content.strip()
            assert len(chunk.content.strip()) > 0


@pytest.fixture
def sample_chunker():
    """Fixture providing a standard chunker for tests."""
    return TextChunker()


@pytest.fixture
def sample_text():
    """Fixture providing sample text for testing."""
    return """This is a sample document for testing the text chunking functionality.

The document contains multiple paragraphs with different types of content.

Some paragraphs are longer and contain more detailed information that might need to be split across multiple chunks.

Other paragraphs are shorter and might fit entirely within a single chunk.

The chunker should handle all these cases appropriately while maintaining semantic coherence."""


@pytest.fixture
def technical_text():
    """Fixture providing technical text for testing."""
    return """API Documentation

The REST API provides endpoints for data retrieval and manipulation.

GET /api/v1/data
Returns JSON data with HTTP status 200 on success.

POST /api/v1/data
Accepts JSON payload and returns created resource.

Error codes:
- 400: Bad Request
- 401: Unauthorized
- 500: Internal Server Error

Use proper authentication headers for all requests."""


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def test_realistic_document_chunking(self, sample_chunker, sample_text):
        """Test chunking a realistic document."""
        chunks = sample_chunker.chunk_text(sample_text, "realistic_doc")
        
        assert len(chunks) >= 1
        
        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == "realistic_doc"
            assert len(chunk.content) > 0
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            
            # Verify metadata
            assert "chunk_size" in chunk.metadata
            assert "word_count" in chunk.metadata
            assert "sentence_count" in chunk.metadata
    
    def test_technical_document_chunking(self, sample_chunker, technical_text):
        """Test chunking technical documentation."""
        chunks = sample_chunker.chunk_text(technical_text, "tech_doc")
        
        assert len(chunks) >= 1
        
        # Should detect technical content
        has_technical = any(
            chunk.metadata.get("appears_technical", False) 
            for chunk in chunks
        )
        assert has_technical
        
        # Should preserve API endpoint information
        api_content = "".join(chunk.content for chunk in chunks)
        assert "/api/v1/data" in api_content
        assert "JSON" in api_content
    
    def test_chunk_statistics_realistic(self, sample_chunker, sample_text):
        """Test statistics generation with realistic content."""
        chunks = sample_chunker.chunk_text(sample_text, "stats_doc")
        stats = sample_chunker.get_chunk_statistics(chunks)
        
        assert stats["total_chunks"] > 0
        assert stats["avg_chunk_size"] > 0
        assert stats["total_characters"] == sum(len(chunk.content) for chunk in chunks)
        assert stats["total_words"] > 0
        assert 0 <= stats["overlap_efficiency"] <= 1