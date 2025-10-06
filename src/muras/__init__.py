__version__ = "0.0.1"

# Re-export everything from submodules for convenient access
from .metrics import evaluate, Sample, QueryContextRelevance
from .embedders import BaseEmbedder, CLIPEmbedder, SentenceTransformerEmbedder, SigLIPEmbedder, SigLIP2Embedder, BLIP2Embedder, ColPaliEmbedder

__all__ = [
    "evaluate",
    "Sample", 
    "QueryContextRelevance",
    "BaseEmbedder",
    "CLIPEmbedder",
    "BLIP2Embedder",
    "SentenceTransformerEmbedder",
    "SigLIPEmbedder",
    "SigLIP2Embedder",
    "ColPaliEmbedder",
]
