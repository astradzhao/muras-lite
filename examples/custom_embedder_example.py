"""
Example: How to create and use a custom embedder

This shows how to implement your own embedder (e.g., BLIP, SigLIP, custom models)
by inheriting from BaseEmbedder.
"""

from typing import List
import torch
from muras import BaseEmbedder, Sample, evaluate


class MyCustomEmbedder(BaseEmbedder):
    """Example custom embedder - replace with your actual implementation."""
    
    def __init__(self, model_name: str = "my-model"):
        super().__init__()
        self.model_name = model_name
        # TODO: Load your model here
        # self.model = load_my_model(model_name)
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using your custom model."""
        # TODO: Implement your text encoding logic
        # For now, return dummy embeddings
        embedding_dim = 512
        embeddings = torch.randn(len(texts), embedding_dim).to(self.device)
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode images using your custom model."""
        # TODO: Implement your image encoding logic
        # For now, return dummy embeddings
        embedding_dim = 512
        embeddings = torch.randn(len(image_paths), embedding_dim).to(self.device)
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings


# Usage example
if __name__ == "__main__":
    # Create samples
    samples = [
        Sample(
            query="What is in this image?",
            contexts_text=["A cat", "A dog", "A bird"],
            contexts_image_paths=["path/to/image1.jpg"],
            answer="The image shows a cat"
        )
    ]
    
    # Option 1: Use default CLIP embedder
    from muras import CLIPEmbedder
    results_clip = evaluate(samples, embedder=CLIPEmbedder())
    print("CLIP Results:", results_clip)
    
    # Option 2: Use your custom embedder
    my_embedder = MyCustomEmbedder(model_name="my-cool-model")
    results_custom = evaluate(samples, embedder=my_embedder)
    print("Custom Results:", results_custom)

