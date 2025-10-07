"""
Qwen2-VL, Qwen2.5-VL generator for multimodal RAG.
"""
from typing import List, Optional
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .base import BaseGenerator, GenerationInput, GenerationOutput

class Qwen2_5VLGenerator(BaseGenerator):
    """
    Qwen2.5-VL generator using transformers.
    
    Supports Qwen2.5-VL models (7B, 3B variants).
    Can use model names:
    - Qwen/Qwen2.5-VL-7B-Instruct
    - Qwen/Qwen2.5-VL-3B-Instruct
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ):
        """
        Initialize Qwen2.5-VL generator.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
        """
        super().__init__(model_name, device)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.processor = None
    
    def _load_model(self):
        """Lazy load Qwen2-VL model."""
        if self.model is not None:
            return
        
        print(f"Loading Qwen2-VL model: {self.model_name}...")
        
        # Prepare quantization config
        kwargs = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif self.load_in_8bit:
            kwargs["load_in_8bit"] = True
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if not (self.load_in_4bit or self.load_in_8bit) else "auto",
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            **kwargs
        )
        
        if not (self.load_in_4bit or self.load_in_8bit) and self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def process_input(self, input_data: GenerationInput) -> str:
        # format all messages
        messages = []
        
        if input_data.system_prompt:
            messages.append({
                "role": "system",
                "content": input_data.system_prompt
            })
        
        content = []
        
        if input_data.context_image_paths:
            for img_path in input_data.context_image_paths:
                content.append({
                    "type": "image",
                    "url": img_path
                })
        
        text_parts = []
        if input_data.context_texts:
            text_parts.append("Context:")
            for i, text in enumerate(input_data.context_texts, 1):
                text_parts.append(f"{i}. {text}")
            text_parts.append("")
        
        text_parts.append(f"Question: {input_data.query}")
        
        content.append({
            "type": "text",
            "text": "\n".join(text_parts)
        })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return text
    
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        """
        Generate response using Qwen2-VL.
        
        Args:
            input_data: GenerationInput with query and context
            
        Returns:
            GenerationOutput with generated text
        """
        self._load_model()
        
        text = self.process_input(input_data)
        
        inputs = self.processor(
            text=[text],
            #images=[input_data.context_image_paths] if input_data.context_image_paths else None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=input_data.max_tokens,
                temperature=input_data.temperature,
                do_sample=input_data.temperature > 0,
            )
        
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        if "assistant\n" in generated_text:
            generated_text = generated_text.split("assistant\n")[-1]
        
        return GenerationOutput(
            generated_text=generated_text.strip(),
            model_name=self.model_name,
            tokens_used=len(output_ids[0]),
            metadata={
                "temperature": input_data.temperature,
                "max_tokens": input_data.max_tokens,
                "num_context_images": len(input_data.context_image_paths),
                "num_context_texts": len(input_data.context_texts)
            }
        )


# class Qwen2VLGenerator(BaseGenerator):
#     """
#     Qwen2-VL generator using transformers.
    
#     Supports Qwen2-VL models (7B, 2B variants).
#     """
    
#     def __init__(
#         self, 
#         model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
#         device: Optional[str] = None,
#         load_in_4bit: bool = False,
#         load_in_8bit: bool = False
#     ):
#         """
#         Initialize Qwen2-VL generator.
        
#         Args:
#             model_name: HuggingFace model name or path
#             device: Device to run on
#             load_in_4bit: Load model in 4-bit quantization
#             load_in_8bit: Load model in 8-bit quantization
#         """
#         super().__init__(model_name, device)
#         self.load_in_4bit = load_in_4bit
#         self.load_in_8bit = load_in_8bit
#         self.model = None
#         self.processor = None
    
#     def _load_model(self):
#         """Lazy load Qwen2-VL model."""
#         if self.model is not None:
#             return
        
#         print(f"Loading Qwen2-VL model: {self.model_name}...")
        
#         # Prepare quantization config
#         kwargs = {}
#         if self.load_in_4bit:
#             from transformers import BitsAndBytesConfig
#             kwargs["quantization_config"] = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.bfloat16
#             )
#         elif self.load_in_8bit:
#             kwargs["load_in_8bit"] = True
        
#         self.processor = AutoProcessor.from_pretrained(
#             self.model_name,
#             trust_remote_code=True
#         )
        
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             self.model_name,
#             torch_dtype=torch.bfloat16 if not (self.load_in_4bit or self.load_in_8bit) else "auto",
#             device_map="auto" if self.device == "cuda" else None,
#             trust_remote_code=True,
#             **kwargs
#         )
        
#         if not (self.load_in_4bit or self.load_in_8bit) and self.device != "cuda":
#             self.model = self.model.to(self.device)
        
#         self.model.eval()
#         print(f"Model loaded on {self.device}")
    
#     def generate(self, input_data: GenerationInput) -> GenerationOutput:
#         """
#         Generate response using Qwen2-VL.
        
#         Args:
#             input_data: GenerationInput with query and context
            
#         Returns:
#             GenerationOutput with generated text
#         """
#         self._load_model()
        
#         # format all messages
#         messages = []
        
#         if input_data.system_prompt:
#             messages.append({
#                 "role": "system",
#                 "content": input_data.system_prompt
#             })
        
#         content = []
        
#         if input_data.context_image_paths:
#             for img_path in input_data.context_image_paths:
#                 content.append({
#                     "type": "image",
#                     "url": img_path
#                 })
        
#         text_parts = []
#         if input_data.context_texts:
#             text_parts.append("Context:")
#             for i, text in enumerate(input_data.context_texts, 1):
#                 text_parts.append(f"{i}. {text}")
#             text_parts.append("")
        
#         text_parts.append(f"Question: {input_data.query}")
        
#         content.append({
#             "type": "text",
#             "text": "\n".join(text_parts)
#         })
        
#         messages.append({
#             "role": "user",
#             "content": content
#         })
        
#         text = self.processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         inputs = self.processor(
#             text=[text],
#             images=[input_data.context_image_paths] if input_data.context_image_paths else None,
#             padding=True,
#             return_tensors="pt"
#         ).to(self.device)
        
#         # generate
#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=input_data.max_tokens,
#                 temperature=input_data.temperature,
#                 do_sample=input_data.temperature > 0,
#             )
        
#         generated_text = self.processor.batch_decode(
#             output_ids,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )[0]
        
#         if "assistant\n" in generated_text:
#             generated_text = generated_text.split("assistant\n")[-1]
        
#         return GenerationOutput(
#             generated_text=generated_text.strip(),
#             model_name=self.model_name,
#             tokens_used=len(output_ids[0]),
#             metadata={
#                 "temperature": input_data.temperature,
#                 "max_tokens": input_data.max_tokens,
#                 "num_context_images": len(input_data.context_image_paths),
#                 "num_context_texts": len(input_data.context_texts)
#             }
#         )

