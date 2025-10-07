# Vector Stores

Vector stores for efficient similarity search in multimodal RAG systems.

## Overview

The vector store module provides abstractions for storing and retrieving document embeddings. It supports:
- **Multimodal documents** (text and/or images)
- **Cross-modal retrieval** (e.g., text query → image results)
- **Metadata filtering**
- **Efficient similarity search**

## Quick Start

```python
from muras import CLIPEmbedder, FAISSVectorStore, Document

# Initialize
embedder = CLIPEmbedder()
vector_store = FAISSVectorStore(embedder=embedder)

# Add documents
documents = [
    Document(id="1", text="Horse racing is a sport"),
    Document(id="2", image_path="path/to/image.jpg"),
]
vector_store.add_documents(documents)

# Search
results = vector_store.search_by_text("racing horses", top_k=5)
for result in results:
    print(f"Score: {result.score:.3f}, Doc: {result.document.id}")
```

## Document Model

```python
Document(
    id: str,                           # Unique identifier
    text: Optional[str] = None,        # Text content
    image_path: Optional[str] = None,  # Path to image
    metadata: Optional[Dict] = None    # Additional metadata
)
```

At least one of `text` or `image_path` must be provided.

## Available Vector Stores

### FAISSVectorStore

FAISS-based vector store with fast similarity search.

**Features:**
- Exact search with `IndexFlatL2`
- Approximate search with `IndexIVFFlat`
- Separate indices for text and images
- Cross-modal search support
- Save/load functionality

**Installation:**
```bash
pip install faiss-cpu  # or faiss-gpu
```

**Usage:**
```python
from muras import FAISSVectorStore, CLIPEmbedder

vector_store = FAISSVectorStore(
    embedder=CLIPEmbedder(),
    index_type="Flat"  # or "IVFFlat" for approximate search
)
```

### LangChainVectorStore

Adapter for any LangChain-supported vector store.

**Supported Backends:**
- **FAISS** - Fast local similarity search 
- **Chroma** - Persistent vector store with rich filtering
- **Pinecone** - Managed cloud vector database
- Planning to add more!

**Installation:**
```bash
# Base requirements
pip install langchain langchain-community

# Backend-specific (choose one or more)
pip install faiss-cpu          # For FAISS
pip install chromadb           # For Chroma
pip install pinecone    # For Pinecone
```

**Usage:**
```python
from muras import LangChainVectorStore, CLIPEmbedder

# FAISS backend
vector_store = LangChainVectorStore(
    embedder=CLIPEmbedder(),
    backend="faiss"
)

# Chroma backend with persistence
vector_store = LangChainVectorStore(
    embedder=CLIPEmbedder(),
    backend="chroma",
    backend_kwargs={"persist_directory": "./my_db"}
)

# Pinecone backend
vector_store = LangChainVectorStore(
    embedder=CLIPEmbedder(),
    backend="pinecone",
    backend_kwargs={
        "index_name": "my-index",
        "environment": "us-west1-gcp"
    }
)
```

## Search Methods

### Text Query Search

```python
results = vector_store.search_by_text(
    query="racing horses",
    top_k=5,
    filter_metadata={"category": "sports"},
    search_type="both"  # "text", "image", or "both"
)
```

### Image Query Search

```python
results = vector_store.search_by_image(
    image_path="path/to/query.jpg",
    top_k=5,
    filter_metadata={"source": "image"},
    search_type="both"
)
```

### Search Types

- `"text"`: Search only text documents
- `"image"`: Search only image documents
- `"both"`: Search all documents (default)

## Metadata Filtering

Filter results by metadata:

```python
results = vector_store.search_by_text(
    query="racing",
    filter_metadata={
        "category": "sports",
        "year": 2024
    }
)
```

Only documents matching ALL filter criteria are returned.

## Document Management

```python
# Add documents
ids = vector_store.add_documents(documents)

# Get document count
count = vector_store.get_document_count()

# Get document by ID
doc = vector_store.get_document_by_id("doc1")

# Delete documents
deleted = vector_store.delete_documents(["doc1", "doc2"])

# Clear all documents
vector_store.clear()
```

## Persistence (FAISS)

```python
# Save to disk
vector_store.save("./my_vector_store")

# Load from disk
vector_store.load("./my_vector_store")
```

## Creating Custom Vector Stores

Extend `BaseVectorStore` to implement custom backends:

```python
from muras import BaseVectorStore, Document, SearchResult

class MyVectorStore(BaseVectorStore):
    def add_documents(self, documents):
        # Your implementation
        pass
    
    def search_by_text(self, query, top_k=5, filter_metadata=None):
        # Your implementation
        pass
    
    def search_by_image(self, image_path, top_k=5, filter_metadata=None):
        # Your implementation
        pass
    
    def delete_documents(self, document_ids):
        # Your implementation
        pass
    
    def clear(self):
        # Your implementation
        pass
    
    def get_document_count(self):
        # Your implementation
        pass
```

## Examples

See `examples/02_vector_store_usage.py` for a complete working example.
