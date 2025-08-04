# models/clip_model.py
import torch
import numpy as np
from PIL import Image
import os
import sys

class CLIPModel:
    def __init__(self, device='cpu'):
        self.device = device
        
        # Available CLIP model variants to try in order
        model_variants = ["ViT-B/16", "ViT-B/32", "RN50"]
        
        # Try to load the CLIP model with better error diagnostics
        self.model = None
        self.clip = None
        self.preprocess = None
        
        try:
            import clip
            self.clip = clip
            
            # Try different model variants
            for model_name in model_variants:
                try:
                    print(f"Attempting to load CLIP model: {model_name}")
                    self.model, self.preprocess = clip.load(model_name, device=device)
                    print(f"Successfully loaded CLIP model: {model_name}")
                    break  # Exit the loop if successful
                except Exception as e:
                    print(f"Failed to load CLIP model {model_name}: {e}")
            
            if self.model is None:
                print("All CLIP model variants failed to load. Using fallback mode.")
                
        except ImportError:
            print("CLIP module not found. Please install with: pip install git+https://github.com/openai/CLIP.git")
            print("Or: pip install ftfy regex tqdm")
            print("Followed by: pip install git+https://github.com/openai/CLIP.git")
            
        except Exception as e:
            print(f"Unexpected error loading CLIP module: {str(e)}")
            print(f"Python path: {sys.path}")
            
    def analyze_image(self, image, text_queries):
        """
        Analyze an image with CLIP against a list of text queries
        
        Args:
            image: RGB image as numpy array
            text_queries: List of text descriptions to compare against
            
        Returns:
            Tensor of similarity scores for each text query
        """
        if self.model is None or self.clip is None or self.preprocess is None:
            # Return balanced scores if model not available
            print("Warning: Using placeholder CLIP scores (model unavailable)")
            return torch.tensor([0.5 for _ in text_queries])
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'))
                
            # Preprocess the image and tokenize the text
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_inputs = self.clip.tokenize(text_queries).to(self.device)
            
            # Get features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
            return similarity[0]  # Return scores for all text queries
            
        except Exception as e:
            print(f"Error during CLIP analysis: {str(e)}")
            return torch.tensor([0.5 for _ in text_queries])