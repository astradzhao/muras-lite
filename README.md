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

### Command Line (Optional)

```bash
# If you want CLI later, you can add it back
# For now, focus on Python API and notebooks
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

For CLIP, you need to install `open-clip-torch`.
For SigLIP, you need to install `protobuf`.
For SentenceTransformer, you need to install `transformers`, `sentence-transformers` and `sentencepiece`.
For ColPali, need to install `colpali-engine>=0.3.0`

## 📊 Metrics

- **QueryContextRelevance**: Similarity between query and retrieved contexts
  - Text context relevance
  - Image context relevance
  - Per-context scores + aggregates

More metrics coming soon!

## 🎯 Use Cases

- Evaluate retrieval quality in multimodal RAG
- Compare different embedders (CLIP, BLIP, SigLIP, ColPali, etc.)
- Analyze per-sample relevance scores
- Debug low-quality retrievals

## 📚 Documentation
- [Examples](examples/README.md) - Code examples
- [Notebooks](notebooks/) - Interactive tutorials

## 🔨 Development

```bash
pip install -e .

# build package
python -m build
```
