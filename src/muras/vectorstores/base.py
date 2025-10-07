"""
Base vector store abstraction for multimodal RAG.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from langchain_core.embeddings import Embeddings

@dataclass
class Document:
    """Represents a document with text and/or image content."""
    
    id: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate that at least one content type is provided."""
        if self.text is None and self.image_path is None:
            raise ValueError("Document must have either text or image_path")


@dataclass
class SearchResult:
    """Represents a search result with similarity score."""
    
    document: Document
    score: float
    embedding: Optional[torch.Tensor] = None

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, embedder):
        """
        Initialize vector store with an embedder.
        
        Args:
            embedder: An instance of BaseEmbedder for encoding documents
        """
        self.embedder = embedder
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs that were added
        """
        pass
    
    @abstractmethod
    def search_by_text(
        self, 
        query: str, 
        top_k_text: int = 5,
        top_k_image: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using a text query.
        
        Args:
            query: Text query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        pass
    
    @abstractmethod
    def search_by_image(
        self, 
        image_path: str, 
        top_k_text: int = 5,
        top_k_image: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using an image query.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> int:
        """
        Delete documents by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        pass
    
    @abstractmethod
    def clear(self):
        """Remove all documents from the vector store."""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the store.
        
        Returns:
            Number of documents
        """
        pass
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Default implementation returns None. Subclasses should override
        this method to query their backend storage for the document.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        return None

class LangChainEmbeddingAdapter(Embeddings):
    """Adapter to make Muras embedders compatible with LangChain."""
    
    def __init__(self, embedder):
        """
        Initialize the adapter.
        
        Args:
            embedder: A Muras BaseEmbedder instance
        """
        super().__init__()
        self.embedder = embedder
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embeddings as lists of floats
        """
        embeddings = self.embedder.encode_text(texts)
        return embeddings.cpu().numpy().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text string
            
        Returns:
            Embedding as a list of floats
        """
        embedding = self.embedder.encode_text([text])
        return embedding[0].cpu().numpy().tolist()
    
    def embed_image(self, image_path: str) -> List[float]:
        """
        Embed a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding as a list of floats
        """
        embedding = self.embedder.encode_images([image_path])
        return embedding[0].cpu().numpy().tolist()
