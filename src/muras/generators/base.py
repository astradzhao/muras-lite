"""
Base generator abstraction for multimodal RAG.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class GenerationInput:
    """Input for generation with retrieved context."""
    
    query: str
    context_texts: List[str] = None
    context_image_paths: List[str] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.context_texts is None:
            self.context_texts = []
        if self.context_image_paths is None:
            self.context_image_paths = []


@dataclass
class GenerationOutput:
    """Output from generation."""
    
    generated_text: str
    model_name: str
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseGenerator(ABC):
    """Abstract base class for VLM generators."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize generator.
        
        Args:
            model_name: Name/path of the model to use
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        self.model_name = model_name
        
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    @abstractmethod
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        """
        Generate text based on query and multimodal context.
        
        Args:
            input_data: GenerationInput with query, texts, and images
            
        Returns:
            GenerationOutput with generated text and metadata
        """
        pass
    
    def format_context(
        self, 
        context_texts: List[str], 
        context_images: List[str]
    ) -> str:
        """
        Format retrieved context into a string for the prompt.
        
        Args:
            context_texts: List of text contexts
            context_images: List of image paths (as references)
            
        Returns:
            Formatted context string
        """
        parts = []
        
        if context_texts:
            parts.append("Text Context:")
            for i, text in enumerate(context_texts, 1):
                parts.append(f"{i}. {text}")
        
        if context_images:
            parts.append("\nImage Context:")
            for i, img_path in enumerate(context_images, 1):
                parts.append(f"{i}. [Image: {img_path}]")
        
        return "\n".join(parts)
    
    def create_prompt(
        self,
        query: str,
        context_texts: List[str],
        context_images: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a complete prompt from query and context.
        
        Args:
            query: User query
            context_texts: Retrieved text contexts
            context_images: Retrieved image paths
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on "
                "the provided context. Use both text and visual information "
                "when available. If the context doesn't contain enough information, "
                "say so clearly."
            )
        
        context = self.format_context(context_texts, context_images)
        
        prompt = f"{system_prompt}\n\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += f"Question: {query}\n\nAnswer:"
        
        return prompt
    
    @abstractmethod
    def _load_model(self):
        """Lazy load the model on first use."""
        pass

