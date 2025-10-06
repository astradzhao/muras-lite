"""
BLIP-based embedder for multimodal RAG tasks.
"""

from typing import List
import torch
from .base import BaseEmbedder
from transformers import AutoProcessor, Blip2TextModelWithProjection, Blip2VisionModelWithProjection

from transformers.image_utils import load_image

import torch.nn.functional as F

# class BLIPEmbedder(BaseEmbedder):
#     """BLIP-based embedder for multimodal retrieval."""
    
#     def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
#         """
#         Initialize BLIP embedder.
        
#         Args:
#             model_name: HuggingFace model identifier.
#         """
#         super().__init__()
#         self.model_name = model_name
#         self.model = None
#         self.processor = None
#         self.text_embedding_dim = None
#         self.image_embedding_dim = None
    
#     def _load_model(self):
#         """Lazy load the BLIP model on first use."""
#         if self.model is None:
#             self.processor = AutoProcessor.from_pretrained(self.model_name)
#             self.model = BlipModel.from_pretrained(self.model_name).to(self.device).eval()
            
#             # Get embedding dimensions using dummy inputs
#             with torch.no_grad():
#                 # Create dummy text input
#                 dummy_text = ["dummy text"]
#                 text_inputs = self.processor(text=dummy_text, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
#                 text_features = self.model.get_text_features(**text_inputs)
#                 self.text_embedding_dim = text_features.shape[1]
#                 print("Text embedding dimension:", self.text_embedding_dim)
                
#                 # Create dummy image input
#                 dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
#                 image_inputs = self.processor(images=[dummy_image], return_tensors="pt").to(self.device)
#                 image_features = self.model.get_image_features(**image_inputs)
#                 self.image_embedding_dim = image_features.shape[1]
#                 print("Image embedding dimension:", self.image_embedding_dim)
    
#     def encode_text(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
#         """
#         Encode text inputs into embeddings.
        
#         Args:
#             texts: List of text strings to encode
#             normalize: Whether to normalize the embeddings
            
#         Returns:
#             Embeddings tensor of shape (len(texts), embedding_dim)
#         """
#         if not texts:
#             return torch.empty(0, self.text_embedding_dim).to(self.device)
        
#         self._load_model()
        
#         with torch.no_grad():
#             inputs = self.processor(text=texts,return_tensors="pt",padding="max_length",truncation=True).to(self.device)
#             text_features = self.model.get_text_features(**inputs)
        
#         return text_features if not normalize else F.normalize(text_features, p=2, dim=-1)
    
#     def encode_images(self, image_paths: List[str], normalize: bool = True) -> torch.Tensor:
#         """
#         Encode images into embeddings.
        
#         Args:
#             image_paths: List of paths to image files
#             normalize: Whether to normalize the embeddings

#         Returns:
#             Embeddings tensor of shape (len(image_paths), embedding_dim)
#         """
#         if not image_paths:
#             return torch.empty(0, self.image_embedding_dim).to(self.device)
        
#         self._load_model()
        
#         images = []
#         for img_path in image_paths:
#             try:
#                 img = load_image(img_path)
#                 images.append(img)
#             except Exception as e:
#                 print(f"Warning: Could not load image {img_path}: {e}")
#                 continue
        
#         if not images:
#             return torch.empty(0, self.image_embedding_dim).to(self.device)

#         inputs = self.processor(images=images,return_tensors="pt").to(self.device)
        
#         with torch.no_grad():
#             image_features = self.model.get_image_features(**inputs)
        
#         return image_features if not normalize else F.normalize(image_features, p=2, dim=-1)

class BLIP2Embedder(BaseEmbedder):
    """Embedder using BLIP2"""
    def __init__(self, model_name: str = "Salesforce/blip2-itm-vit-g"):
        """
        Initialize BLIP embedder.
        
        Args:
            model_name: HuggingFace model identifier.
        """
        super().__init__()
        self.model_name = model_name
        self.text_model = None
        self.vision_model = None
        self.processor = None
        self.text_embedding_dim = None
        self.image_embedding_dim = None
    
    def _load_model(self):
        """Lazy load the BLIP model on first use."""
        if self.text_model is None or self.vision_model is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.text_model = Blip2TextModelWithProjection.from_pretrained(self.model_name).to(self.device).eval()
            self.vision_model = Blip2VisionModelWithProjection.from_pretrained(self.model_name).to(self.device).eval()
            
            self.embedding_dim = self.text_model.config.image_text_hidden_size
            print("Embedding dimension:", self.embedding_dim)

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
            return torch.empty(0, self.embedding_dim).to(self.device)
        
        self._load_model()
        
        with torch.inference_mode():
            inputs = self.processor(text=texts,return_tensors="pt",padding="max_length",truncation=True).to(self.device)
            text_features = self.text_model(**inputs).text_embeds
            text_features = text_features.mean(dim=1)
        
        print("Text features shape:", text_features.shape)
        
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

        inputs = self.processor(images=images,return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            image_features = self.vision_model(**inputs).image_embeds
            image_features = image_features.mean(dim=1)
        
        print("Image features shape:", image_features.shape)

        return image_features if not normalize else F.normalize(image_features, p=2, dim=-1)