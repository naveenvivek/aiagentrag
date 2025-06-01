"""
RAG Pipeline implementation for document ingestion, vectorization, and retrieval.
"""
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

from src.documents.processor import DocumentProcessor
from src.vectorstore.chroma_store import ChromaVectorStore
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger("rag.pipeline")


class RAGPipeline:
    """Complete RAG pipeline for document processing and retrieval."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the RAG pipeline.
        
        Args:
            collection_name: Name of the vector store collection
        """
        self.document_processor = DocumentProcessor()
        self.vector_store = ChromaVectorStore(collection_name)
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        logger.info("RAG pipeline initialized")
    
    def add_document_from_file(self, file_path: str) -> List[str]:
        """
        Process a single file and add it to the RAG system.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of chunk IDs added to the vector store
        """
        try:
            # Process the document
            doc_data = self.document_processor.process_file(file_path)
            
            # Chunk the document
            chunks = self._chunk_text(doc_data['content'], doc_data)
            
            # Add chunks to vector store
            chunk_ids = self.vector_store.add_documents(chunks)
            
            logger.info(f"Successfully added {len(chunks)} chunks from {file_path}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def add_documents_from_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """
        Process all supported files in a directory and add them to the RAG system.
        
        Args:
            directory_path: Path to the directory containing documents
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of all chunk IDs added to the vector store
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_chunk_ids = []
        supported_extensions = self.document_processor.supported_formats
        
        # Get file pattern based on recursive flag
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    chunk_ids = self.add_document_from_file(str(file_path))
                    all_chunk_ids.extend(chunk_ids)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Processed directory {directory_path}, added {len(all_chunk_ids)} total chunks")
        return all_chunk_ids
    
    def add_text_content(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add raw text content directly to the RAG system.
        
        Args:
            content: Raw text content to add
            metadata: Optional metadata for the content
            
        Returns:
            List of chunk IDs added to the vector store
        """
        base_metadata = metadata or {}
        base_metadata.update({
            'content_type': 'raw_text',
            'size': len(content)
        })
        
        # Chunk the content
        chunks = self._chunk_text(content, base_metadata)
        
        # Add to vector store
        chunk_ids = self.vector_store.add_documents(chunks)
        
        logger.info(f"Added {len(chunks)} chunks from raw text content")
        return chunk_ids
    
    def add_web_content(self, url: str) -> List[str]:
        """
        Fetch content from a web URL and add it to the RAG system.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            List of chunk IDs added to the vector store
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            metadata = {
                'source_url': url,
                'content_type': 'web_content',
                'size': len(content)
            }
            
            return self.add_text_content(content, metadata)
            
        except Exception as e:
            logger.error(f"Error fetching web content from {url}: {str(e)}")
            raise
    
    def search_documents(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using the query.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant document chunks with metadata
        """
        return self.vector_store.similarity_search(query, k, filter_metadata)
    
    def get_context_for_query(self, query: str, k: int = 5) -> str:
        """
        Get formatted context for a query to use in generation.
        
        Args:
            query: Search query
            k: Number of document chunks to retrieve
            
        Returns:
            Formatted context string
        """
        documents = self.search_documents(query, k)
        
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source_info = ""
            if doc['metadata'].get('file_name'):
                source_info = f" (Source: {doc['metadata']['file_name']})"
            elif doc['metadata'].get('source_url'):
                source_info = f" (Source: {doc['metadata']['source_url']})"
            
            context_parts.append(f"[Context {i}]{source_info}:\n{doc['content']}")
        
        return "\n\n".join(context_parts)
    
    def _chunk_text(self, text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        Args:
            text: Text to chunk
            base_metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries
        """
        if not text:
            return []
        
        chunks = []
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        
        # Simple text chunking strategy
        words = text.split()
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'chunk_size': len(chunk_words),
                'start_word_index': i,
                'end_word_index': min(i + chunk_size, len(words))
            })
            
            chunks.append({
                'content': chunk_text,
                **chunk_metadata
            })
        
        return chunks
    
    def delete_documents_by_source(self, source_identifier: str) -> None:
        """
        Delete all documents from a specific source.
        
        Args:
            source_identifier: File path or URL to identify documents to delete
        """
        # This would require implementing a way to query by metadata
        # For now, we'll log this as a feature to implement
        logger.warning("Delete by source not yet implemented")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline."""
        return self.vector_store.get_collection_stats()
    
    def clear_all_documents(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.clear_collection()
        logger.info("Cleared all documents from RAG pipeline")
