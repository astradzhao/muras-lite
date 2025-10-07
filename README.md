# Muras: Multimodal RAG Assessment Suite

A pluggable evaluation framework for multimodal RAG systems. Inspired by https://github.com/explodinggradients/ragas. This is currently just a personal project.

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage (Python)

```python
from muras import Sample, evaluate, CLIPEmbedder

# Create evaluation samples
samples = [
    Sample(
        query="What animals are racing?",
        contexts_text=["Horse racing is a sport", "Dogs are pets"],
        contexts_image_paths=["path/to/image.jpg"],
        answer="Horses are racing"
    )
]

# Evaluate similarities with CLIP embedder
clip_embedder = CLIPEmbedder()
results = evaluate(samples, embedder=clip_embedder)
print(results['aggregate'])
```
```


## 📁 Repository Structure

```
muras/
├── src/muras/              # Core package
│   ├── embedders/          # Pluggable embedders (CLIP, BLIP, SigLIP, ColPali, custom)
│   └── metrics.py          # Evaluation metrics
├── examples/               # Python scripts
│   ├── 01_compare_embedders.py
├── notebooks/              # Jupyter notebooks
└── tests/                  # Tests
```

## 🔌 Pluggable Architecture

### Custom Embedders

Create custom embedders by inheriting from `BaseEmbedder`:

```python
from muras import BaseEmbedder, evaluate

class MyEmbedder(BaseEmbedder):
    def encode_text(self, texts):
        # Your implementation
        pass
    
    def encode_images(self, image_paths):
        # Your implementation
        pass

# Use it
results = evaluate(samples, embedder=MyEmbedder())
```

See `examples/custom_embedder_example.py` for details.

### Vector Stores

Use vector stores for efficient multimodal retrieval:

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

# Search with text query
results = vector_store.search_by_text("racing horses", top_k=5)

# Search with image query
results = vector_store.search_by_image("query.jpg", top_k=5)
```

**Use any LangChain vector store:**

```python
from muras import LangChainVectorStore, CLIPEmbedder

# Works with FAISS, Chroma, Pinecone, Weaviate, Qdrant, and more!
vector_store = LangChainVectorStore(
    embedder=CLIPEmbedder(),
    backend="chroma",  # or "faiss", "pinecone", etc.
    backend_kwargs={"persist_directory": "./my_db"}
)
```

See `examples/02_vector_store_usage.py` and `examples/03_langchain_vectorstores.py` for complete examples.

## 📦 Optional Dependencies

**Embedders:**
- **CLIP**: `pip install open-clip-torch`
- **SigLIP**: `pip install protobuf`
- **SentenceTransformers**: `pip install transformers sentence-transformers sentencepiece`
- **ColPali**: `pip install colpali-engine>=0.3.0`

**Vector Stores:**
- **FAISS**: `pip install faiss-cpu` (or `faiss-gpu`)
- **LangChain Integration**: `pip install langchain langchain-community`
- **Chroma**: `pip install chromadb`
- **Pinecone**: `pip install pinecone-client`

See `examples/04_end_to_end_rag.py` on how to run a full multimodal RAG solution on some data!

## 📊 Metrics

- **QueryContextRelevance**: Similarity between query and retrieved contexts
  - Text context relevance
  - Image context relevance
  - Per-context scores + aggregates

More metrics coming soon.

## 🎯 Use Cases

- Evaluate retrieval quality in multimodal RAG
- Compare different embedders (CLIP, BLIP, SigLIP, ColPali, etc.)
- Analyze per-sample relevance scores
- Debug low-quality retrievals

## 📚 Documentation
- [Examples](examples/) - Code examples
- [Notebooks](notebooks/) - Interactive tutorials

## 🔨 Development

```bash
pip install -e .

# build package
python -m build
```
