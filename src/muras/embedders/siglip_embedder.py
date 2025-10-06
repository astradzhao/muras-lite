"""
SigLIP-based embedder for multimodal RAG tasks.
"""

from typing import List
import torch
from .base import BaseEmbedder
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from transformers.image_utils import load_image

from PIL import Image
import torch.nn.functional as F


class SigLIPEmbedder(BaseEmbedder):
    """SigLIP-based embedder for multimodal retrieval."""
    
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        """
        Initialize SigLIP embedder.
        
        Args:
            model_name: HuggingFace model identifier. Options include:
                - "google/siglip-base-patch16-224"
                - "google/siglip-large-patch16-256"
                - "google/siglip-so400m-patch14-384"
                Other options can be found at https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba
        """
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.text_embedding_dim = None
        self.image_embedding_dim = None
    
    def _load_model(self):
        """Lazy load the SigLIP model on first use."""
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            self.text_embedding_dim = self.model.text_model.config.hidden_size
            self.image_embedding_dim = self.model.vision_model.config.hidden_size
    
    def encode_text(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode text inputs into embeddings.
        
        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize the embeddings
            
        Returns:
            Embeddings tensor of shape (len(texts), embedding_dim)
        """
        if not texts:
            return torch.empty(0, self.text_embedding_dim).to(self.device)
        
        self._load_model()
        
        with torch.no_grad():
            inputs = self.tokenizer(texts,return_tensors="pt",padding="max_length",truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
        
        return text_features if not normalize else F.normalize(text_features, p=2, dim=-1)
    
    def encode_images(self, image_paths: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode images into embeddings.
        
        Args:
            image_paths: List of paths to image files
            normalize: Whether to normalize the embeddings

        Returns:
            Embeddings tensor of shape (len(image_paths), embedding_dim)
        """
        if not image_paths:
            return torch.empty(0, self.image_embedding_dim).to(self.device)
        
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
            return torch.empty(0, self.image_embedding_dim).to(self.device)

        inputs = self.processor(images=images,return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features if not normalize else F.normalize(image_features, p=2, dim=-1)

    
class SigLIP2Embedder(BaseEmbedder):
    """SigLIP 2-based embedder for multimodal retrieval."""

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        """
        Initialize SigLIP 2 embedder. Options can be found at https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107
        """
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.text_embedding_dim = None
        self.image_embedding_dim = None

    def _load_model(self):
        """Lazy load the SigLIP 2 model and processor."""
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Infer embedding dimension from the model
            self.text_embedding_dim = self.model.text_model.config.hidden_size
            self.image_embedding_dim = self.model.vision_model.config.hidden_size

    def encode_text(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode text inputs into SigLIP 2 embeddings.

        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize the embeddings

        Returns:
            Embeddings tensor of shape (len(texts), embedding_dim)
        """
        self._load_model()
        if not texts:
            return torch.empty(0, self.text_embedding_dim).to(self.device)
        
        inputs = self.tokenizer(texts,return_tensors="pt",padding=True,truncation=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features if not normalize else F.normalize(text_features, p=2, dim=-1)

    def encode_images(self, image_paths: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode image paths into SigLIP 2 embeddings.

        Args:
            image_paths: List of strings for image paths
            normalize: Whether to normalize the embeddings

        Returns:
            Embeddings tensor of shape (len(image_paths), embedding_dim)
        """
        self._load_model()
        if not image_paths:
            return torch.empty(0, self.image_embedding_dim).to(self.device)

        images = []
        for img_path in image_paths:
            try:
                img = load_image(img_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
        if not images:
            return torch.empty(0, self.image_embedding_dim).to(self.device)

        inputs = self.processor(images=images,return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features if not normalize else F.normalize(image_features, p=2, dim=-1)

