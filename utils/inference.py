"""
Model loading and inference utility for financial report summarization.
Handles loading the fine-tuned model and generating summaries.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os


# System prompt for financial summarization
SYSTEM_PROMPT = """You are a financial analyst. Summarize the following financial report concisely, focusing on key financial metrics, performance highlights, and strategic outlook."""


class FinancialSummarizer:
    """
    Wrapper class for the fine-tuned financial summarization model.
    """
    
    def __init__(self, model_path: str, base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the summarizer with the fine-tuned model.
        
        Args:
            model_path: Path to the fine_tuned_model.pt file
            base_model_name: Name of the base model from HuggingFace
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """
        Load the fine-tuned model and tokenizer.
        """
        print(f"Loading model from {self.model_path}...")
        print(f"Using device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # For CPU, skip quantization to avoid bitsandbytes issues
            if self.device == "cpu":
                print("Using CPU mode - skipping quantization")
                # Load base model without quantization for CPU
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                )
            else:
                # Configure quantization for memory efficiency (GPU only)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                # Load base model with quantization
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            
            # Load the fine-tuned checkpoint
            # Note: weights_only=False is needed for models with custom configs
            # Only use this if you trust the source of the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Load state dict into model
            base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            self.model = base_model
            self.model.eval()
            
            print("✓ Model loaded successfully!")
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def format_prompt(self, document: str, max_length: int = 3000) -> str:
        """
        Format the document into an instruction-style prompt.
        
        Args:
            document: Financial report text
            max_length: Maximum character length for the document
            
        Returns:
            Formatted prompt string
        """
        # Truncate document if too long
        truncated_doc = document[:max_length] if len(document) > max_length else document
        
        prompt = f"""### Instruction:
{SYSTEM_PROMPT}

### Document:
{truncated_doc}

### Summary:"""
        
        return prompt
    
    def generate_summary(self, document: str, max_new_tokens: int = 128) -> str:
        """
        Generate a summary for the given financial document.
        
        Args:
            document: Financial report text
            max_new_tokens: Maximum number of tokens to generate (reduced for CPU)
            
        Returns:
            Generated summary text
        """
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        try:
            print("Formatting prompt...")
            # Format the prompt
            prompt = self.format_prompt(document)
            
            print("Tokenizing input...")
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # Reduced for faster processing
            ).to(self.model.device)
            
            print("Generating summary...")
            # Generate summary with CPU-optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=0.7,
                    early_stopping=True,
                    use_cache=True,
                )
            
            print("Decoding output...")
            # Decode the output
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the summary part (after "### Summary:")
            if "### Summary:" in output_text:
                summary = output_text.split("### Summary:")[-1].strip()
            else:
                summary = output_text.strip()
            
            print("Summary generated successfully!")
            return summary
            
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")


def load_summarizer(model_path: str) -> FinancialSummarizer:
    """
    Convenience function to load and initialize the summarizer.
    
    Args:
        model_path: Path to the fine_tuned_model.pt file
        
    Returns:
        Initialized FinancialSummarizer instance
    """
    summarizer = FinancialSummarizer(model_path)
    summarizer.load_model()
    return summarizer
