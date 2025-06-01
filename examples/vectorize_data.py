#!/usr/bin/env python3
"""
Example script demonstrating how to vectorize data and add it to the RAG system.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.utils.logging import get_logger

logger = get_logger("examples.vectorize_data")


def main():
    """Demonstrate various ways to vectorize and add data to RAG."""
    
    # Initialize the RAG pipeline
    rag = RAGPipeline(collection_name="example_documents")
    
    print("🚀 RAG Data Vectorization Examples")
    print("=" * 50)
    
    # Example 1: Add text content directly
    print("\n1. Adding raw text content...")
    sample_text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any device that perceives its 
    environment and takes actions that maximize its chance of achieving its goals.
    """
    
    text_metadata = {
        "source": "manual_entry",
        "topic": "artificial_intelligence",
        "author": "example"
    }
    
    chunk_ids = rag.add_text_content(sample_text, text_metadata)
    print(f"✅ Added {len(chunk_ids)} chunks from raw text")
    
    # Example 2: Add documents from a file (if it exists)
    print("\n2. Adding documents from files...")
    
    # Create a sample document for demonstration
    sample_doc_path = "data/sample_document.txt"
    os.makedirs("data", exist_ok=True)
    
    with open(sample_doc_path, "w") as f:
        f.write("""
        Machine Learning is a subset of artificial intelligence that provides systems the ability 
        to automatically learn and improve from experience without being explicitly programmed. 
        Machine learning focuses on the development of computer programs that can access data 
        and use it to learn for themselves.
        
        The process of learning begins with observations or data, such as examples, direct experience, 
        or instruction, in order to look for patterns in data and make better decisions in the future 
        based on the examples that we provide.
        """)
    
    try:
        file_chunk_ids = rag.add_document_from_file(sample_doc_path)
        print(f"✅ Added {len(file_chunk_ids)} chunks from file: {sample_doc_path}")
    except Exception as e:
        print(f"❌ Error adding file: {e}")
    
    # Example 3: Add web content
    print("\n3. Adding web content...")
    try:
        # Note: This requires internet connection and may fail
        web_chunk_ids = rag.add_web_content("https://en.wikipedia.org/wiki/Artificial_intelligence")
        print(f"✅ Added {len(web_chunk_ids)} chunks from web content")
    except Exception as e:
        print(f"❌ Error adding web content: {e}")
    
    # Example 4: Search and retrieve
    print("\n4. Searching for relevant content...")
    query = "What is machine learning?"
    
    search_results = rag.search_documents(query, k=3)
    print(f"🔍 Found {len(search_results)} relevant documents for query: '{query}'")
    
    for i, result in enumerate(search_results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result['content'][:200]}...")
        print(f"Metadata: {result['metadata']}")
        if result.get('distance'):
            print(f"Distance: {result['distance']:.4f}")
    
    # Example 5: Get formatted context for generation
    print("\n5. Getting formatted context for LLM...")
    context = rag.get_context_for_query(query, k=2)
    print("📝 Formatted context:")
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Example 6: Pipeline statistics
    print("\n6. Pipeline statistics...")
    stats = rag.get_pipeline_stats()
    print(f"📊 Collection: {stats['collection_name']}")
    print(f"📚 Total documents: {stats['document_count']}")
    print(f"🤖 Embedding model: {stats['embedding_model']}")


def advanced_examples():
    """More advanced examples for data vectorization."""
    
    print("\n" + "=" * 50)
    print("🎯 Advanced Vectorization Examples")
    print("=" * 50)
    
    rag = RAGPipeline(collection_name="advanced_example")
    
    # Example: Batch processing multiple documents
    print("\n1. Batch processing multiple text contents...")
    
    documents = [
        {
            "content": "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            "metadata": {"topic": "nlp", "type": "definition"}
        },
        {
            "content": "Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.",
            "metadata": {"topic": "computer_vision", "type": "definition"}
        },
        {
            "content": "Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "metadata": {"topic": "deep_learning", "type": "definition"}
        }
    ]
    
    all_chunk_ids = []
    for doc in documents:
        chunk_ids = rag.add_text_content(doc["content"], doc["metadata"])
        all_chunk_ids.extend(chunk_ids)
    
    print(f"✅ Added {len(all_chunk_ids)} chunks from {len(documents)} documents")
    
    # Example: Filtering search results by metadata
    print("\n2. Searching with metadata filters...")
    
    # Search only in NLP-related content
    nlp_results = rag.search_documents(
        "What is natural language processing?",
        k=5,
        filter_metadata={"topic": "nlp"}
    )
    print(f"🔍 Found {len(nlp_results)} NLP-specific results")
    
    # Search in all definition-type content
    definition_results = rag.search_documents(
        "artificial intelligence definition",
        k=5,
        filter_metadata={"type": "definition"}
    )
    print(f"🔍 Found {len(definition_results)} definition-type results")


if __name__ == "__main__":
    try:
        main()
        advanced_examples()
        
        print("\n" + "=" * 50)
        print("✨ All examples completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in examples: {e}")
        print(f"❌ Error: {e}")
