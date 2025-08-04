"""
Text chunking utilities for splitting documents into manageable chunks.

This module implements recursive character text splitter with configurable chunk size and overlap,
semantic coherence preservation logic, and chunk metadata generation.

Implements requirements 3.1, 3.2, 3.3, 3.4.
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.models.schemas import DocumentChunk


@dataclass
class ChunkingConfig:
    """Configuration for text chunking parameters."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = None
    keep_separator: bool = True
    
    def __post_init__(self):
        if self.separators is None:
            # Default separators in order of preference for semantic coherence
            self.separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation sentences
                "? ",    # Question sentences
                "; ",    # Semicolon breaks
                ", ",    # Comma breaks
                " ",     # Word breaks
                ""       # Character level (fallback)
            ]


class TextChunker:
    """
    Recursive character text splitter with semantic coherence preservation.
    
    Implements configurable chunk size and overlap with intelligent splitting
    that maintains context within chunks and preserves document structure.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the text chunker with configuration.
        
        Args:
            config: Chunking configuration. If None, uses default settings.
        """
        self.config = config or ChunkingConfig()
        
        # Validate configuration
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if self.config.chunk_size < 50:
            raise ValueError("Chunk size must be at least 50 characters")
    
    def chunk_text(
        self, 
        text: str, 
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Split text into chunks with semantic coherence preservation.
        
        Args:
            text: The text to be chunked
            document_id: Identifier of the parent document
            metadata: Additional metadata to include with chunks
            
        Returns:
            List of DocumentChunk objects with proper metadata
            
        Raises:
            ValueError: If text is empty or document_id is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty")
        
        # Clean and normalize text
        cleaned_text = self._preprocess_text(text)
        
        # Split text recursively using separators
        chunks = self._recursive_split(cleaned_text)
        
        # Create DocumentChunk objects with metadata
        document_chunks = []
        current_position = 0
        
        for i, chunk_content in enumerate(chunks):
            # Find the actual position in original text
            start_char, end_char = self._find_chunk_position(
                text, chunk_content, current_position
            )
            
            # Generate chunk metadata
            chunk_metadata = self._generate_chunk_metadata(
                chunk_content, i, len(chunks), start_char, end_char, metadata
            )
            
            # Create DocumentChunk
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char
            )
            
            document_chunks.append(chunk)
            current_position = end_char
        
        return document_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text to normalize whitespace and remove artifacts.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        # Clean up common document artifacts first
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Remove excessive line breaks but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Normalize spaces within lines but preserve line breaks
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Normalize whitespace within each line
            normalized_line = re.sub(r'\s+', ' ', line).strip()
            if normalized_line:  # Only keep non-empty lines
                normalized_lines.append(normalized_line)
            elif normalized_lines and normalized_lines[-1]:  # Preserve paragraph breaks
                normalized_lines.append('')
        
        # Join lines back together
        text = '\n'.join(normalized_lines)
        
        # Clean up multiple consecutive empty lines
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Strip leading/trailing whitespace
        return text.strip()
    
    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using separators to maintain semantic coherence.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order of preference
        for separator in self.config.separators:
            if separator in text:
                chunks = self._split_by_separator(text, separator)
                if chunks:
                    return chunks
        
        # Fallback to character-level splitting if no separator works
        return self._split_by_character(text)
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text by a specific separator with overlap handling.
        
        Args:
            text: Text to split
            separator: Separator to use for splitting
            
        Returns:
            List of text chunks, or empty list if splitting doesn't help
        """
        if separator == "":
            return self._split_by_character(text)
        
        # Split by separator
        splits = text.split(separator)
        
        # If we only get one split, this separator doesn't help
        if len(splits) == 1:
            return []
        
        # Reconstruct with separator if keep_separator is True
        if self.config.keep_separator and separator != " ":
            splits = [split + separator for split in splits[:-1]] + [splits[-1]]
        
        # Combine splits into chunks with overlap
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If adding this split would exceed chunk size
            if len(current_chunk) + len(split) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + split
                else:
                    # Split is too large, need to split it further
                    sub_chunks = self._recursive_split(split)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
            else:
                current_chunk += split
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_by_character(self, text: str) -> List[str]:
        """
        Split text at character level as fallback method.
        
        Args:
            text: Text to split
            
        Returns:
            List of character-level chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            
            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Look for word boundary within last 50 characters
                search_start = max(end - 50, start)
                word_boundary = text.rfind(' ', search_start, end)
                
                if word_boundary > start:
                    end = word_boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.config.chunk_overlap)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from the end of a chunk.
        
        Args:
            text: Text to extract overlap from
            
        Returns:
            Overlap text for next chunk
        """
        if len(text) <= self.config.chunk_overlap:
            return text
        
        overlap_start = len(text) - self.config.chunk_overlap
        overlap_text = text[overlap_start:]
        
        # Try to start overlap at a word boundary
        space_pos = overlap_text.find(' ')
        if space_pos > 0:
            overlap_text = overlap_text[space_pos + 1:]
        
        return overlap_text
    
    def _find_chunk_position(
        self, 
        original_text: str, 
        chunk_content: str, 
        start_search: int
    ) -> Tuple[int, int]:
        """
        Find the position of chunk content in the original text.
        
        Args:
            original_text: Original document text
            chunk_content: Content of the chunk
            start_search: Position to start searching from
            
        Returns:
            Tuple of (start_char, end_char) positions
        """
        # Clean chunk content for searching
        search_content = chunk_content.strip()
        
        # Find the chunk in original text starting from start_search
        pos = original_text.find(search_content, start_search)
        
        if pos != -1:
            return pos, pos + len(search_content)
        
        # Fallback: approximate position based on chunk index
        return start_search, start_search + len(chunk_content)
    
    def _generate_chunk_metadata(
        self,
        chunk_content: str,
        chunk_index: int,
        total_chunks: int,
        start_char: int,
        end_char: int,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for a chunk.
        
        Args:
            chunk_content: Content of the chunk
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            start_char: Starting character position
            end_char: Ending character position
            base_metadata: Base metadata to extend
            
        Returns:
            Dictionary containing chunk metadata
        """
        metadata = base_metadata.copy() if base_metadata else {}
        
        # Basic chunk information
        metadata.update({
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_size': len(chunk_content),
            'start_char': start_char,
            'end_char': end_char,
            'word_count': len(chunk_content.split()),
            'sentence_count': len(re.findall(r'[.!?]+', chunk_content)),
        })
        
        # Semantic analysis
        metadata.update(self._analyze_chunk_semantics(chunk_content))
        
        # Position information
        metadata['is_first_chunk'] = chunk_index == 0
        metadata['is_last_chunk'] = chunk_index == total_chunks - 1
        metadata['relative_position'] = chunk_index / max(total_chunks - 1, 1)
        
        return metadata
    
    def _analyze_chunk_semantics(self, content: str) -> Dict[str, Any]:
        """
        Analyze semantic properties of chunk content.
        
        Args:
            content: Chunk content to analyze
            
        Returns:
            Dictionary with semantic analysis results
        """
        analysis = {}
        
        # Text structure analysis
        # Look for lines that are all caps or title case and standalone
        analysis['has_headers'] = bool(
            re.search(r'^[A-Z][A-Z\s:]+$', content, re.MULTILINE) or
            re.search(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*:?\s*$', content, re.MULTILINE) or
            re.search(r'^\s*[A-Z][A-Z\s]+\s*$', content, re.MULTILINE)
        )
        analysis['has_lists'] = bool(re.search(r'^\s*[-*•]\s+', content, re.MULTILINE))
        analysis['has_numbers'] = bool(re.search(r'\d+', content))
        analysis['has_dates'] = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content))
        
        # Content type hints
        analysis['appears_technical'] = bool(re.search(r'\b(API|HTTP|JSON|XML|SQL)\b', content, re.IGNORECASE))
        analysis['appears_legal'] = bool(re.search(r'\b(shall|whereas|hereby|pursuant)\b', content, re.IGNORECASE))
        analysis['appears_financial'] = bool(re.search(r'[$€£¥]\d+|\b\d+\.\d{2}\b', content))
        
        # Language complexity
        avg_word_length = sum(len(word) for word in content.split()) / max(len(content.split()), 1)
        analysis['avg_word_length'] = round(avg_word_length, 2)
        analysis['complexity_score'] = min(avg_word_length / 5.0, 1.0)  # Normalized complexity
        
        return analysis
    
    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Generate statistics about the chunking results.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {'total_chunks': 0}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        word_counts = [chunk.metadata.get('word_count', 0) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'total_words': sum(word_counts),
            'avg_words_per_chunk': sum(word_counts) / len(word_counts) if word_counts else 0,
            'overlap_efficiency': self._calculate_overlap_efficiency(chunks)
        }
    
    def _calculate_overlap_efficiency(self, chunks: List[DocumentChunk]) -> float:
        """
        Calculate the efficiency of overlap between chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Overlap efficiency score between 0 and 1
        """
        if len(chunks) < 2:
            return 1.0
        
        total_overlap = 0
        overlap_count = 0
        
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].content
            next_chunk = chunks[i + 1].content
            
            # Find common text at the end of current and start of next
            overlap = self._find_text_overlap(current_chunk, next_chunk)
            total_overlap += len(overlap)
            overlap_count += 1
        
        expected_overlap = self.config.chunk_overlap * overlap_count
        if expected_overlap == 0:
            return 1.0
        
        return min(total_overlap / expected_overlap, 1.0)
    
    def _find_text_overlap(self, text1: str, text2: str) -> str:
        """
        Find overlapping text between two chunks.
        
        Args:
            text1: First text (end will be checked for overlap)
            text2: Second text (beginning will be checked for overlap)
            
        Returns:
            Overlapping text
        """
        max_overlap = min(len(text1), len(text2), self.config.chunk_overlap)
        
        for i in range(max_overlap, 0, -1):
            if text1[-i:] == text2[:i]:
                return text1[-i:]
        
        return ""


def create_chunker(chunk_size: int = 1000, chunk_overlap: int = 200) -> TextChunker:
    """
    Factory function to create a TextChunker with custom configuration.
    
    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        Configured TextChunker instance
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return TextChunker(config)


def chunk_document_text(
    text: str,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None
) -> List[DocumentChunk]:
    """
    Convenience function to chunk document text with default settings.
    
    Args:
        text: Text to be chunked
        document_id: Identifier of the parent document
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        metadata: Additional metadata to include with chunks
        
    Returns:
        List of DocumentChunk objects
    """
    chunker = create_chunker(chunk_size, chunk_overlap)
    return chunker.chunk_text(text, document_id, metadata)