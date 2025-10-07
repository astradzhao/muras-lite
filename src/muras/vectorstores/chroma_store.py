"""
Chroma-based vector store implementation using LangChain.
"""
from typing import List, Dict, Any, Optional
from .base import BaseVectorStore, Document, SearchResult, LangChainEmbeddingAdapter


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma vector store powered by LangChain.
    
    Features:
    - Persistent storage with automatic saves
    - Rich metadata filtering capabilities
    - Good performance for medium-to-large datasets
    - Separate collections for text and images
    """
    
    def __init__(
        self, 
        embedder,
        persist_directory: str = "./chroma_db",
        collection_name: str = "muras"
    ):
        """
        Initialize Chroma vector store.
        
        Args:
            embedder: A Muras BaseEmbedder instance
            persist_directory: Directory for persisting the vector store
            collection_name: Base name for collections (will create text_ and image_ variants)
        """
        super().__init__(embedder)
        
        try:
            from langchain.vectorstores import Chroma
        except ImportError:
            raise ImportError(
                "LangChain and Chroma are required for ChromaVectorStore. "
                "Install with: pip install langchain chromadb"
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_adapter = LangChainEmbeddingAdapter(embedder)
        
        # Initialize Chroma collections
        self.text_store = Chroma(
            collection_name=f"{collection_name}_text",
            embedding_function=self.embedding_adapter,
            persist_directory=persist_directory
        )
        
        self.image_store = Chroma(
            collection_name=f"{collection_name}_image",
            embedding_function=self.embedding_adapter,
            persist_directory=persist_directory
        )
        
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
        
        from langchain.schema import Document as LangChainDocument
        
        added_ids = []
        text_docs = [doc for doc in documents if doc.text is not None]
        image_docs = [doc for doc in documents if doc.image_path is not None]
        
        # Process text documents
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
            
            self.text_store.add_documents(lc_docs)
            added_ids.extend([doc.id for doc in text_docs])
        
        # Process image documents
        if image_docs:
            lc_docs = []
            for doc in image_docs:
                metadata = doc.metadata or {}
                metadata["muras_id"] = doc.id
                metadata["content_type"] = "image"
                metadata["image_path"] = doc.image_path
                
                lc_docs.append(LangChainDocument(
                    page_content=f"[IMAGE: {doc.image_path}]",
                    metadata=metadata
                ))
            
            # Add with pre-computed embeddings
            image_embeddings = [
                self.embedding_adapter.embed_image(doc.image_path)
                for doc in image_docs
            ]
            
            self.image_store.add_texts(
                texts=[doc.page_content for doc in lc_docs],
                metadatas=[doc.metadata for doc in lc_docs],
                embeddings=image_embeddings
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
        
        # Remove Muras-specific metadata
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
            top_k_text: Number of text results to return
            top_k_image: Number of image results to return
            filter_metadata: Optional metadata filters (fully supported by Chroma)
            search_type: "text" (text only), "image" (image only), or "both" (default)
            
        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        text_results = []
        image_results = []
        
        # Build Chroma filter
        where_filter = None
        if filter_metadata:
            # Chroma uses a specific filter format
            where_filter = {k: {"$eq": v} for k, v in filter_metadata.items()}
        
        if search_type in ["text", "both"]:
            try:
                lc_results = self.text_store.similarity_search_with_score(
                    query,
                    k=top_k_text,
                    filter=where_filter
                )
                text_results = self._convert_results(lc_results)
            except Exception as e:
                print(f"Warning: Text search failed: {e}")
        
        if search_type in ["image", "both"]:
            try:
                lc_results = self.image_store.similarity_search_with_score(
                    query,
                    k=top_k_image,
                    filter=where_filter
                )
                image_results = self._convert_results(lc_results)
            except Exception as e:
                print(f"Warning: Image search failed: {e}")
        
        # Apply additional metadata filtering if needed
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
            top_k_text: Number of text results to return (cross-modal)
            top_k_image: Number of image results to return
            filter_metadata: Optional metadata filters (fully supported by Chroma)
            search_type: "text" (text only), "image" (image only), or "both" (default)
            
        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        query_embedding = self.embedding_adapter.embed_image(image_path)
        text_results = []
        image_results = []
        
        # Build Chroma filter
        where_filter = None
        if filter_metadata:
            where_filter = {k: {"$eq": v} for k, v in filter_metadata.items()}
        
        if search_type in ["text", "both"]:
            try:
                # Chroma doesn't have similarity_search_by_vector_with_relevance_scores
                # We'll use the collection's query method directly
                lc_results = self.text_store._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k_text,
                    where=where_filter
                )
                
                # Convert Chroma results to our format
                if lc_results and lc_results['ids']:
                    from langchain.schema import Document as LangChainDocument
                    for i in range(len(lc_results['ids'][0])):
                        lc_doc = LangChainDocument(
                            page_content=lc_results['documents'][0][i],
                            metadata=lc_results['metadatas'][0][i]
                        )
                        distance = lc_results['distances'][0][i]
                        text_results.extend(self._convert_results([(lc_doc, distance)]))
            except Exception as e:
                print(f"Warning: Cross-modal text search failed: {e}")
        
        if search_type in ["image", "both"]:
            try:
                lc_results = self.image_store._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k_image,
                    where=where_filter
                )
                
                if lc_results and lc_results['ids']:
                    from langchain.schema import Document as LangChainDocument
                    for i in range(len(lc_results['ids'][0])):
                        lc_doc = LangChainDocument(
                            page_content=lc_results['documents'][0][i],
                            metadata=lc_results['metadatas'][0][i]
                        )
                        distance = lc_results['distances'][0][i]
                        image_results.extend(self._convert_results([(lc_doc, distance)]))
            except Exception as e:
                print(f"Warning: Image search failed: {e}")
        
        # Apply additional metadata filtering if needed
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
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID from the backend storage.
        
        Chroma supports efficient ID-based retrieval with metadata filtering.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        # Try text store
        try:
            text_results = self.text_store.similarity_search(
                "",
                k=1,
                filter={"muras_id": {"$eq": document_id}}
            )
            if text_results:
                return self._langchain_to_muras_document(text_results[0])
        except:
            pass
        
        # Try image store
        try:
            image_results = self.image_store.similarity_search(
                "",
                k=1,
                filter={"muras_id": {"$eq": document_id}}
            )
            if image_results:
                return self._langchain_to_muras_document(image_results[0])
        except:
            pass
        
        return None
    
    def delete_documents(self, document_ids: List[str]) -> int:
        """
        Delete documents by their IDs.
        
        Chroma supports efficient deletion by metadata filter.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        deleted_count = 0
        
        for doc_id in document_ids:
            try:
                self.text_store._collection.delete(
                    where={"muras_id": {"$eq": doc_id}}
                )
                deleted_count += 1
                continue
            except:
                pass
            
            try:
                self.image_store._collection.delete(
                    where={"muras_id": {"$eq": doc_id}}
                )
                deleted_count += 1
            except:
                pass
        
        self._doc_count = max(0, self._doc_count - deleted_count)
        return deleted_count
    
    def clear(self):
        """
        Remove all documents from the vector store.
        
        Note: This deletes the collections and recreates them.
        """
        try:
            # Delete collections
            self.text_store._collection.delete()
            self.image_store._collection.delete()
            
            # Recreate collections
            from langchain.vectorstores import Chroma
            self.text_store = Chroma(
                collection_name=f"{self.collection_name}_text",
                embedding_function=self.embedding_adapter,
                persist_directory=self.persist_directory
            )
            
            self.image_store = Chroma(
                collection_name=f"{self.collection_name}_image",
                embedding_function=self.embedding_adapter,
                persist_directory=self.persist_directory
            )
            
            self._doc_count = 0
        except Exception as e:
            print(f"Warning: Failed to clear collections: {e}")
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the store.
        
        Returns:
            Number of unique documents
        """
        return self._doc_count
    
    def persist(self):
        """
        Persist the vector store to disk.
        
        Note: Chroma automatically persists changes, but calling this
        ensures all pending changes are written to disk.
        """
        try:
            self.text_store.persist()
            self.image_store.persist()
        except:
            pass
    
    @classmethod
    def load(cls, embedder, persist_directory: str, collection_name: str = "muras"):
        """
        Load a persisted vector store from disk.
        
        Args:
            embedder: A Muras BaseEmbedder instance
            persist_directory: Directory where the store was persisted
            collection_name: Base name for collections
            
        Returns:
            Loaded ChromaVectorStore instance
        """
        instance = cls(
            embedder,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        try:
            text_count = instance.text_store._collection.count()
            image_count = instance.image_store._collection.count()
            instance._doc_count = text_count + image_count
        except:
            instance._doc_count = 0
        
        return instance
