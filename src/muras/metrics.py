from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import open_clip
from PIL import Image
import torch
import numpy as np

class Sample(BaseModel):
    query: str
    contexts_text: List[str] = []
    contexts_image_paths: List[str] = []
    answer: str
    reference: Optional[str] = None


class QueryContextRelevance:
    """CLIP-based metric for measuring query-context relevance."""
    
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self.model.eval()
            self.model.to(self.device)
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
    
    def compute_text_similarities(self, query: str, texts: List[str]) -> List[float]:
        """Compute CLIP similarity between query and each text context."""
        if not texts:
            return []
        
        self._load_model()
        
        with torch.no_grad():
            query_tokens = self.tokenizer([query]).to(self.device)
            query_features = self.model.encode_text(query_tokens)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            
            text_tokens = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarities = (query_features @ text_features.T).squeeze(0)
            
        return similarities.cpu().tolist() if len(texts) > 1 else [similarities.item()]
    
    def compute_image_similarities(self, query: str, image_paths: List[str]) -> List[float]:
        """Compute CLIP similarity between query and each image context."""
        if not image_paths:
            return []
        
        self._load_model()
        
        with torch.no_grad():
            query_tokens = self.tokenizer([query]).to(self.device)
            query_features = self.model.encode_text(query_tokens)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            
            images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    continue
            
            if not images:
                return []
            
            image_batch = torch.stack(images).to(self.device)
            image_features = self.model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarities = (query_features @ image_features.T).squeeze(0)
            
        return similarities.cpu().tolist() if len(images) > 1 else [similarities.item()]
    
    def score(self, sample: Sample) -> Dict[str, Any]:
        """Compute CLIP scores for both text and image contexts."""
        text_similarities = self.compute_text_similarities(sample.query, sample.contexts_text)
        image_similarities = self.compute_image_similarities(sample.query, sample.contexts_image_paths)
        
        return {
            "text_similarities": text_similarities,
            "image_similarities": image_similarities,
            "text_avg": float(np.mean(text_similarities)) if text_similarities else 0.0,
            "image_avg": float(np.mean(image_similarities)) if image_similarities else 0.0,
            "text_max": float(np.max(text_similarities)) if text_similarities else 0.0,
            "image_max": float(np.max(image_similarities)) if image_similarities else 0.0,
        }

def evaluate(samples: List[Sample]) -> Dict[str, Any]:
    """Evaluate samples using CLIP-based relevance metrics."""
    metric = QueryContextRelevance()
    scores = [metric.score(s) for s in samples]
    
    all_text_avg = [s["text_avg"] for s in scores]
    all_image_avg = [s["image_avg"] for s in scores]
    
    return {
        "samples": scores,
        "aggregate": {
            "text_relevance_avg": float(np.mean(all_text_avg)) if all_text_avg else 0.0,
            "image_relevance_avg": float(np.mean(all_image_avg)) if all_image_avg else 0.0,
        },
        "n_samples": len(scores),
    }
