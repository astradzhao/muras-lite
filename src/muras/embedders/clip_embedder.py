from typing import List
import torch
import open_clip
from PIL import Image
from .base import BaseEmbedder


class CLIPEmbedder(BaseEmbedder):
    """CLIP-based embedder using OpenCLIP."""
    
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self.model.eval()
            self.model.to(self.device)
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using CLIP text encoder."""
        if not texts:
            return torch.empty(0, 512).to(self.device)  # 512 is CLIP embedding dim
        
        self._load_model()
        
        with torch.no_grad():
            text_tokens = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
        
        return text_features
    
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode images using CLIP image encoder."""
        if not image_paths:
            return torch.empty(0, 512).to(self.device)
        
        self._load_model()
        
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(self.preprocess(img))
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                continue
        
        if not images:
            return torch.empty(0, 512).to(self.device)
        
        with torch.no_grad():
            image_batch = torch.stack(images).to(self.device)
            image_features = self.model.encode_image(image_batch)
        
        return image_features

