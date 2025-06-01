"""
Vector store implementation using ChromaDB.
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

from src.utils.config import settings as app_settings
from src.utils.logging import get_logger

logger = get_logger("vectorstore.chroma")


class ChromaVectorStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection to use
        """
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(app_settings.embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=app_settings.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            texts.append(doc['content'])
            
            # Prepare metadata (ChromaDB doesn't support nested objects)
            metadata = {
                'file_name': doc.get('file_name', ''),
                'file_path': doc.get('file_path', ''),
                'extension': doc.get('extension', ''),
                'size': doc.get('size', 0),
                'chunk_index': doc.get('chunk_index', 0)
            }
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            documents.append(doc)
        
        logger.info(f"Found {len(documents)} similar documents for query")
        return documents
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from vector store")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'embedding_model': app_settings.embedding_model
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete the collection and recreate it
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared collection: {self.collection_name}")
