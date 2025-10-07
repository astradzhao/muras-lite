"""
FAISS-based vector store implementation using LangChain.
"""
from typing import List, Dict, Any, Optional
from .base import BaseVectorStore, Document, SearchResult, LangChainEmbeddingAdapter
from langchain.schema import Document as LangChainDocument
from langchain_community.vectorstores import FAISS

import os

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS vector store powered by LangChain.
    """
    
    def __init__(
        self, 
        embedder,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedder: A Muras BaseEmbedder instance
            persist_directory: Optional directory for persisting the vector store
        """
        super().__init__(embedder)
        
        self.persist_directory = persist_directory
        self.embedding_adapter = LangChainEmbeddingAdapter(embedder)
        self.text_store = None
        self.image_store = None
        self._doc_count = 0
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
        
        added_ids = []
        text_docs = [doc for doc in documents if doc.text is not None]
        image_docs = [doc for doc in documents if doc.image_path is not None]
        
        if text_docs:
            lc_docs = []
            for doc in text_docs:
                metadata = doc.metadata or {}
                metadata["muras_id"] = doc.id
                metadata["content_type"] = "text"
                metadata["text_content"] = doc.text
                
                lc_docs.append(LangChainDocument(
                    page_content=doc.text,
                    metadata=metadata
                ))
            
            if self.text_store is None:
                self.text_store = FAISS.from_documents(
                    lc_docs,
                    self.embedding_adapter
                )
            else:
                self.text_store.add_documents(lc_docs)
            
            added_ids.extend([doc.id for doc in text_docs])
        
        if image_docs:
            lc_docs = []
            image_embeddings = []
            
            for doc in image_docs:
                metadata = doc.metadata or {}
                metadata["muras_id"] = doc.id
                metadata["content_type"] = "image"
                metadata["image_path"] = doc.image_path
                
                lc_docs.append(LangChainDocument(
                    page_content=f"[IMAGE: {doc.image_path}]",
                    metadata=metadata
                ))
                
                image_embeddings.append(
                    self.embedding_adapter.embed_image(doc.image_path)
                )
            
            if self.image_store is None:
                texts = [doc.page_content for doc in lc_docs]
                metadatas = [doc.metadata for doc in lc_docs]
                self.image_store = FAISS.from_embeddings(
                    text_embeddings=list(zip(texts, image_embeddings)),
                    embedding=self.embedding_adapter,
                    metadatas=metadatas
                )
            else:
                texts = [doc.page_content for doc in lc_docs]
                metadatas = [doc.metadata for doc in lc_docs]
                self.image_store.add_embeddings(
                    list(zip(texts, image_embeddings)),
                    metadatas=metadatas
                )
            
            added_ids.extend([doc.id for doc in image_docs if doc.id not in added_ids])
        
        self._doc_count += len(added_ids)
        return added_ids
    
    def _langchain_to_muras_document(self, lc_doc) -> Optional[Document]:
        """Convert a LangChain document back to a Muras Document."""
        metadata = lc_doc.metadata
        doc_id = metadata.get("muras_id")
        if not doc_id:
            return None
        
        content_type = metadata.get("content_type", "text")
        
        clean_metadata = {k: v for k, v in metadata.items() 
                         if k not in ["muras_id", "content_type", "text_content", "image_path"]}
        
        if content_type == "text":
            return Document(
                id=doc_id,
                text=metadata.get("text_content") or lc_doc.page_content,
                metadata=clean_metadata if clean_metadata else None
            )
        else:
            return Document(
                id=doc_id,
                image_path=metadata.get("image_path"),
                metadata=clean_metadata if clean_metadata else None
            )
    
    def _convert_results(self, langchain_results: List[tuple]) -> List[SearchResult]:
        """Convert LangChain search results to Muras SearchResult objects."""
        results = []
        
        for lc_doc, score in langchain_results:
            doc = self._langchain_to_muras_document(lc_doc)
            if not doc:
                continue
            
            # Convert distance to similarity (lower distance = higher similarity)
            similarity = 1.0 / (1.0 + score) if score >= 0 else abs(score)
            
            results.append(SearchResult(
                document=doc,
                score=float(similarity),
                embedding=None
            ))
        
        return results
    
    def search_by_text(
        self, 
        query: str, 
        top_k_text: int = 5,
        top_k_image: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        search_type: str = "both"
    ) -> List[SearchResult]:
        """
        Search for similar documents using a text query.
        
        Args:
            query: Text query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (not fully supported by FAISS)
            search_type: "text" (text only), "image" (image only), or "both" (default)
            
        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        text_results = []
        image_results = []
        
        if search_type in ["text", "both"] and self.text_store is not None:
            try:
                lc_results = self.text_store.similarity_search_with_score(
                    query,
                    k=top_k_text
                )
                text_results.extend(self._convert_results(lc_results))
            except Exception as e:
                print(f"Warning: Text search failed: {e}")
        
        if search_type in ["image", "both"] and self.image_store is not None:
            try:
                lc_results = self.image_store.similarity_search_with_score(
                    query,
                    k=top_k_image
                )
                image_results.extend(self._convert_results(lc_results))
            except Exception as e:
                print(f"Warning: Image search failed: {e}")
        
        # Apply metadata filtering if provided (manual filtering)
        if filter_metadata:
            text_results = [
                r for r in text_results 
                if r.document.metadata and all(
                    r.document.metadata.get(k) == v 
                    for k, v in filter_metadata.items()
                )
            ]

            image_results = [
                r for r in image_results 
                if r.document.metadata and all(
                    r.document.metadata.get(k) == v 
                    for k, v in filter_metadata.items()
                )
            ]
        
        text_results.sort(key=lambda x: x.score, reverse=True)
        image_results.sort(key=lambda x: x.score, reverse=True)

        results = text_results + image_results
        return results
    
    def search_by_image(
        self, 
        image_path: str, 
        top_k_text: int = 5,
        top_k_image: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        search_type: str = "both"
    ) -> List[SearchResult]:
        """
        Search for similar documents using an image query.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (not fully supported by FAISS)
            search_type: "text" (text only), "image" (image only), or "both" (default)
            
        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        query_embedding = self.embedding_adapter.embed_image(image_path)
        text_results = []
        image_results = []
        
        if search_type in ["text", "both"] and self.text_store is not None:
            try:
                lc_results = self.text_store.similarity_search_with_score_by_vector(
                    query_embedding,
                    k=top_k_text
                )
                text_results.extend(self._convert_results(lc_results))
            except Exception as e:
                print(f"Warning: Cross-modal text search failed: {e}")
        
        if search_type in ["image", "both"] and self.image_store is not None:
            try:
                lc_results = self.image_store.similarity_search_with_score_by_vector(
                    query_embedding,
                    k=top_k_image
                )
                image_results.extend(self._convert_results(lc_results))
            except Exception as e:
                print(f"Warning: Image search failed: {e}")
        
        # Apply metadata filtering if provided (manual filtering)
        if filter_metadata:
            text_results = [
                r for r in image_results 
                if r.document.metadata and all(
                    r.document.metadata.get(k) == v 
                    for k, v in filter_metadata.items()
                )
            ]
        
            image_results = [
                r for r in image_results 
                if r.document.metadata and all(
                    r.document.metadata.get(k) == v 
                    for k, v in filter_metadata.items()
                )
            ]

        text_results.sort(key=lambda x: x.score, reverse=True)
        image_results.sort(key=lambda x: x.score, reverse=True)
        
        results = text_results + image_results
        return results
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID from the backend storage.
        
        Note: FAISS doesn't support efficient ID lookups with filters.
        This performs a similarity search with a filter, which may be slow.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        # Try text store
        if self.text_store is not None:
            try:
                # FAISS doesn't support filters well, so we search and filter manually
                docs = self.text_store.docstore._dict.values()
                for doc in docs:
                    if doc.metadata.get("muras_id") == document_id:
                        return self._langchain_to_muras_document(doc)
            except:
                pass
        
        # Try image store
        if self.image_store is not None:
            try:
                docs = self.image_store.docstore._dict.values()
                for doc in docs:
                    if doc.metadata.get("muras_id") == document_id:
                        return self._langchain_to_muras_document(doc)
            except:
                pass
        
        return None
    
    def delete_documents(self, document_ids: List[str]) -> int:
        """
        Delete documents by their IDs.
        
        Note: FAISS doesn't support efficient deletion. This implementation
        rebuilds the indices without the deleted documents.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        deleted_count = 0
        doc_ids_set = set(document_ids)
        
        # Rebuild text store
        if self.text_store is not None:
            try:
                all_docs = list(self.text_store.docstore._dict.values())
                remaining_docs = [
                    doc for doc in all_docs 
                    if doc.metadata.get("muras_id") not in doc_ids_set
                ]
                deleted_count += len(all_docs) - len(remaining_docs)
                
                if remaining_docs:
                    self.text_store = FAISS.from_documents(
                        remaining_docs,
                        self.embedding_adapter
                    )
                else:
                    self.text_store = None
            except Exception as e:
                print(f"Warning: Failed to delete from text store: {e}")
        
        # Rebuild image store
        if self.image_store is not None:
            try:
                all_docs = list(self.image_store.docstore._dict.values())
                remaining_docs = [
                    doc for doc in all_docs 
                    if doc.metadata.get("muras_id") not in doc_ids_set
                ]
                
                if remaining_docs:
                    # Re-embed images
                    embeddings = []
                    for doc in remaining_docs:
                        img_path = doc.metadata.get("image_path")
                        if img_path:
                            embeddings.append(self.embedding_adapter.embed_image(img_path))
                    
                    texts = [doc.page_content for doc in remaining_docs]
                    metadatas = [doc.metadata for doc in remaining_docs]
                    self.image_store = FAISS.from_embeddings(
                        text_embeddings=list(zip(texts, embeddings)),
                        embedding=self.embedding_adapter,
                        metadatas=metadatas
                    )
                else:
                    self.image_store = None
            except Exception as e:
                print(f"Warning: Failed to delete from image store: {e}")
        
        self._doc_count = max(0, self._doc_count - deleted_count)
        return deleted_count
    
    def clear(self):
        """Remove all documents from the vector store."""
        self.text_store = None
        self.image_store = None
        self._doc_count = 0
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the store.
        
        Returns:
            Number of unique documents
        """
        return self._doc_count
    
    def save(self, path: Optional[str] = None):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the store (uses persist_directory if not provided)
        """        
        save_path = path or self.persist_directory
        if not save_path:
            raise ValueError("No path provided and persist_directory not set")
        
        os.makedirs(save_path, exist_ok=True)
        
        if self.text_store:
            self.text_store.save_local(os.path.join(save_path, "text_store"))
        
        if self.image_store:
            self.image_store.save_local(os.path.join(save_path, "image_store"))
    
    @classmethod
    def load(cls, embedder, path: str):
        """
        Load a persisted vector store from disk.
        
        Args:
            embedder: A Muras BaseEmbedder instance
            path: Directory path to load the store from
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        
        instance = cls(embedder, persist_directory=path)
        
        text_path = os.path.join(path, "text_store")
        if os.path.exists(text_path):
            instance.text_store = FAISS.load_local(
                text_path,
                instance.embedding_adapter,
                allow_dangerous_deserialization=True
            )
        
        image_path = os.path.join(path, "image_store")
        if os.path.exists(image_path):
            instance.image_store = FAISS.load_local(
                image_path,
                instance.embedding_adapter,
                allow_dangerous_deserialization=True
            )
        
        # Count documents
        count = 0
        if instance.text_store:
            count += len(instance.text_store.docstore._dict)
        if instance.image_store:
            count += len(instance.image_store.docstore._dict)
        instance._doc_count = count
        
        return instance
