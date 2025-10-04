"""
Sentence-Transformers based embedder for text and image embeddings.

This embedder uses sentence-transformers which provides a unified API
for text and image embeddings with various pre-trained models.
"""

from typing import List
import torch
from .base import BaseEmbedder
from PIL import Image
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-Transformers based embedder for unified text/image encoding."""
    
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        super().__init__()
        self.model_name = model_name
        self.model = None
    
    def _load_model(self):
        """Lazy load the sentence-transformers model on first use."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)

    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using sentence-transformers."""
        if not texts:
            return torch.empty(0, 512).to(self.device)  # Common embedding dim
        
        self._load_model()
        
        with torch.no_grad():
            text_features = self.model.encode(texts, convert_to_tensor=True)
            text_features = text_features.to(self.device)
        
        return text_features
    
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode images using sentence-transformers."""
        if not image_paths:
            return torch.empty(0, 512).to(self.device)
        
        self._load_model()
        
        with torch.no_grad():
            images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
            image_features = self.model.encode(images, convert_to_tensor=True)

            image_features = image_features.to(self.device)
        
        return image_features
