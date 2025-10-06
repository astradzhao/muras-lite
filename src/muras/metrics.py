from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from .embedders import BaseEmbedder
import numpy as np

class Sample(BaseModel):
    query: str
    contexts_text: List[str] = []
    contexts_image_paths: List[str] = []
    answer: str
    reference: Optional[str] = None


class QueryContextRelevance:
    """Generic metric for measuring query-context relevance using any embedder."""
    
    def __init__(self, embedder: BaseEmbedder = None):
        """
        Initialize the metric with a specific embedder.
        
        Args:
            embedder: Any embedder inheriting from BaseEmbedder. 
        """
        if embedder is None:
            from .embedders import CLIPEmbedder
            embedder = CLIPEmbedder()
        self.embedder = embedder
    
    def compute_text_similarities(self, query: str, texts: List[str]) -> List[float]:
        """Compute similarity between query and each text context."""
        if not texts:
            return []
        
        query_embeddings = self.embedder.encode_text([query])
        text_embeddings = self.embedder.encode_text(texts)
        
        similarities = self.embedder.compute_cosine_similarity(query_embeddings, text_embeddings)
        print(f"Similarities: {similarities}")
        similarities = similarities.squeeze(0)  # Remove query dimension
        
        return similarities.cpu().tolist() if len(texts) > 1 else [similarities.item()]
    
    def compute_image_similarities(self, query: str, image_paths: List[str]) -> List[float]:
        """Compute similarity between query and each image context."""
        if not image_paths:
            return []
        
        query_embeddings = self.embedder.encode_text([query])
        image_embeddings = self.embedder.encode_images(image_paths)
        
        if image_embeddings.shape[0] == 0:
            return []
        
        similarities = self.embedder.compute_cosine_similarity(query_embeddings, image_embeddings)
        similarities = similarities.squeeze(0)
        
        return similarities.cpu().tolist() if image_embeddings.shape[0] > 1 else [similarities.item()]
    
    def score(self, sample: Sample) -> Dict[str, Any]:
        """Compute similarity scores for both text and image contexts."""
        text_similarities = self.compute_text_similarities(sample.query, sample.contexts_text)
        image_similarities = self.compute_image_similarities(sample.query, sample.contexts_image_paths)

        print(f"Text similarities: {text_similarities}")
        print(f"Image similarities: {image_similarities}")
        print("-" * 60)
        
        return {
            "text_similarities": text_similarities,
            "image_similarities": image_similarities,
            "text_avg": float(np.mean(text_similarities)) if text_similarities else 0.0,
            "image_avg": float(np.mean(image_similarities)) if image_similarities else 0.0,
            "text_max": float(np.max(text_similarities)) if text_similarities else 0.0,
            "image_max": float(np.max(image_similarities)) if image_similarities else 0.0,
        }

def evaluate(samples: List[Sample], embedder: BaseEmbedder = None) -> Dict[str, Any]:
    """
    Evaluate samples using query-context relevance metrics.
    
    Args:
        samples: List of Sample objects to evaluate
        embedder: Optional embedder to use. Defaults to CLIPEmbedder.
        
    Returns:
        Dictionary with per-sample scores and aggregate metrics
    """
    import numpy as np
    
    metric = QueryContextRelevance(embedder=embedder)
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
