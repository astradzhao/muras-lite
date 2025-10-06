from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
import torch.nn.functional as F

if TYPE_CHECKING:
    import torch


class BaseEmbedder(ABC):
    """Abstract base class for multimodal embedders."""
    
    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def encode_text(self, texts: List[str]) -> "torch.Tensor":
        """
        Encode text inputs into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Normalized embeddings tensor of shape (len(texts), embedding_dim)
        """
        pass
    
    @abstractmethod
    def encode_images(self, image_paths: List[str]) -> "torch.Tensor":
        """
        Encode images into embeddings.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Normalized embeddings tensor of shape (len(image_paths), embedding_dim)
        """
        pass
    
    def compute_cosine_similarity(self, embeddings1: "torch.Tensor", embeddings2: "torch.Tensor") -> "torch.Tensor":
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: Tensor of shape (n, embedding_dim)
            embeddings2: Tensor of shape (m, embedding_dim)
            
        Returns:
            Similarity matrix of shape (n, m)
        """
        embeddings1_norm = F.normalize(embeddings1, p=2, dim=-1)
        embeddings2_norm = F.normalize(embeddings2, p=2, dim=-1)
        
        similarity = embeddings1_norm @ embeddings2_norm.T

        print(f"Similarity: {similarity}")
        return similarity

