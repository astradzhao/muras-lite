"""
ColPali-based embedder for multimodal RAG tasks.
"""
from typing import List
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers.image_utils import load_image
from .base import BaseEmbedder


class ColPaliEmbedder(BaseEmbedder):
    """ColPali-based embedder using ColPali."""
    
    def __init__(self, model_name = "vidore/colpali-v1.3"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.embedding_dim = None # typically 128, same for text and image
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            self.model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(self.model_name)
            self.embedding_dim = 128
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using CLIP text encoder."""
        if not texts:
            return torch.empty(0, self.embedding_dim).to(self.device)
        
        self._load_model()

        with torch.no_grad():
            text_tokens = self.processor.process_queries(texts).to(self.device)
            text_features = self.model(**text_tokens)
            text_features = text_features.mean(dim=1)
        
        return text_features
    
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode images using CLIP image encoder."""
        if not image_paths:
            return torch.empty(0, self.embedding_dim).to(self.device)
        
        self._load_model()
        
        images = []
        for img_path in image_paths:
            try:
                img = load_image(img_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                continue
        
        if not images:
            return torch.empty(0, self.embedding_dim).to(self.device)

        batch_images = self.processor.process_images(images).to(self.device)
        
        with torch.no_grad():
            image_features = self.model(**batch_images)
            image_features = image_features.mean(dim=1)
        
        return image_features

