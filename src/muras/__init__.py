__version__ = "0.0.1"

# Re-export everything from submodules for convenient access
from .metrics import evaluate, Sample, QueryContextRelevance
from .embedders import BaseEmbedder, CLIPEmbedder, SentenceTransformerEmbedder, SigLIPEmbedder, SigLIP2Embedder

__all__ = [
    "evaluate",
    "Sample", 
    "QueryContextRelevance",
    "BaseEmbedder",
    "CLIPEmbedder",
    "SentenceTransformerEmbedder",
    "SigLIPEmbedder",
    "SigLIP2Embedder",
]
