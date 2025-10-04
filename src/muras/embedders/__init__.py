# Simple direct imports - no need for complex lazy loading here
from .base import BaseEmbedder
from .clip_embedder import CLIPEmbedder
from .sentence_transformer_embedder import SentenceTransformerEmbedder
from .siglip_embedder import SigLIPEmbedder, SigLIP2Embedder

__all__ = ["BaseEmbedder", "CLIPEmbedder", "SentenceTransformerEmbedder", "SigLIPEmbedder", "SigLIP2Embedder"]

