#!/usr/bin/env python3
"""
Utility script for adding different types of data to the RAG system.
Usage examples:
    python add_to_rag.py --text "Your text content here"
    python add_to_rag.py --file /path/to/document.pdf
    python add_to_rag.py --directory /path/to/documents/
    python add_to_rag.py --url https://example.com/article
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.utils.logging import get_logger

logger = get_logger("scripts.add_to_rag")


def add_text_to_rag(text: str, collection_name: str = "documents", metadata: dict = None):
    """Add text content to RAG system."""
    rag = RAGPipeline(collection_name)
    chunk_ids = rag.add_text_content(text, metadata)
    print(f"✅ Added {len(chunk_ids)} chunks from text content")
    return chunk_ids


def add_file_to_rag(file_path: str, collection_name: str = "documents"):
    """Add a file to RAG system."""
    rag = RAGPipeline(collection_name)
    chunk_ids = rag.add_document_from_file(file_path)
    print(f"✅ Added {len(chunk_ids)} chunks from file: {file_path}")
    return chunk_ids


def add_directory_to_rag(directory_path: str, collection_name: str = "documents", recursive: bool = True):
    """Add all files in a directory to RAG system."""
    rag = RAGPipeline(collection_name)
    chunk_ids = rag.add_documents_from_directory(directory_path, recursive)
    print(f"✅ Added {len(chunk_ids)} chunks from directory: {directory_path}")
    return chunk_ids


def add_url_to_rag(url: str, collection_name: str = "documents"):
    """Add web content to RAG system."""
    rag = RAGPipeline(collection_name)
    chunk_ids = rag.add_web_content(url)
    print(f"✅ Added {len(chunk_ids)} chunks from URL: {url}")
    return chunk_ids


def search_rag(query: str, collection_name: str = "documents", k: int = 5):
    """Search the RAG system."""
    rag = RAGPipeline(collection_name)
    results = rag.search_documents(query, k)
    
    print(f"🔍 Search results for: '{query}'")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result['content'][:200]}...")
        print(f"Source: {result['metadata'].get('file_name', result['metadata'].get('source_url', 'Unknown'))}")
        if result.get('distance'):
            print(f"Relevance score: {1 - result['distance']:.4f}")


def get_rag_stats(collection_name: str = "documents"):
    """Get RAG system statistics."""
    rag = RAGPipeline(collection_name)
    stats = rag.get_pipeline_stats()
    
    print("📊 RAG System Statistics:")
    print("-" * 30)
    print(f"Collection: {stats['collection_name']}")
    print(f"Total documents: {stats['document_count']}")
    print(f"Embedding model: {stats['embedding_model']}")


def main():
    parser = argparse.ArgumentParser(description="Add data to RAG system")
    parser.add_argument("--text", help="Text content to add")
    parser.add_argument("--file", help="File path to add")
    parser.add_argument("--directory", help="Directory path to add (all supported files)")
    parser.add_argument("--url", help="URL to fetch and add content from")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--stats", action="store_true", help="Show RAG statistics")
    parser.add_argument("--collection", default="documents", help="Collection name (default: documents)")
    parser.add_argument("--k", type=int, default=5, help="Number of search results (default: 5)")
    parser.add_argument("--recursive", action="store_true", default=True, help="Recursive directory search")
    
    args = parser.parse_args()
    
    if not any([args.text, args.file, args.directory, args.url, args.search, args.stats]):
        parser.print_help()
        return
    
    try:
        if args.text:
            add_text_to_rag(args.text, args.collection)
        
        if args.file:
            add_file_to_rag(args.file, args.collection)
        
        if args.directory:
            add_directory_to_rag(args.directory, args.collection, args.recursive)
        
        if args.url:
            add_url_to_rag(args.url, args.collection)
        
        if args.search:
            search_rag(args.search, args.collection, args.k)
        
        if args.stats:
            get_rag_stats(args.collection)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
