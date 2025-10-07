__version__ = "0.0.1"

# Re-export everything from submodules for convenient access
from .metrics import evaluate, Sample, QueryContextRelevance
from .embedders import (
    BaseEmbedder, 
    CLIPEmbedder, 
    SentenceTransformerEmbedder, 
    SigLIPEmbedder, 
    SigLIP2Embedder, 
    BLIP2Embedder, 
    ColPaliEmbedder
)
from .vectorstores import (
    BaseVectorStore, 
    Document, 
    SearchResult, 
    FAISSVectorStore,
    ChromaVectorStore,
    #PineconeVectorStore,
    LangChainEmbeddingAdapter
)
from .generators import (
    BaseGenerator,
    GenerationInput,
    GenerationOutput,
    Qwen2_5VLGenerator,
    # SmolVLMGenerator,
    # GemmaGenerator,
    # Gemma3Generator
)

__all__ = [
    # Metrics
    "evaluate",
    "Sample", 
    "QueryContextRelevance",
    # Embedders
    "BaseEmbedder",
    "CLIPEmbedder",
    "Qwen2_5VLGenerator",
    "BLIP2Embedder",
    "SentenceTransformerEmbedder",
    "SigLIPEmbedder",
    "SigLIP2Embedder",
    "ColPaliEmbedder",
    # Vector Stores
    "BaseVectorStore",
    "Document",
    "SearchResult",
    "FAISSVectorStore",
    "ChromaVectorStore",
    #"PineconeVectorStore",
    "LangChainEmbeddingAdapter",
    # Generators
    "BaseGenerator",
    "GenerationInput",
    "GenerationOutput",
    "Qwen2VLGenerator",
    # "SmolVLMGenerator",
    # "GemmaGenerator",
    # "Gemma3Generator",
]
