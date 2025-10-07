"""
Generator implementations for multimodal RAG.
"""
from .base import BaseGenerator, GenerationInput, GenerationOutput
from .qwen_generator import Qwen2_5VLGenerator
# from .smolvlm_generator import SmolVLMGenerator
# from .gemma_generator import GemmaGenerator, Gemma3Generator

__all__ = [
    "BaseGenerator",
    "GenerationInput",
    "GenerationOutput",
    "Qwen2_5VLGenerator",
    # "SmolVLMGenerator",
    # "GemmaGenerator",
    # "Gemma3Generator",
]

