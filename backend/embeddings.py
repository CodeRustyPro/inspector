"""
CLIP embedding service.
Generates 512-dimensional vectors from images using openai/clip-vit-base-patch32.
"""
import io
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Union

MODEL_NAME = "openai/clip-vit-base-patch32"

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        print(f"Loading CLIP model ({MODEL_NAME})...")
        _model = CLIPModel.from_pretrained(MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model.eval()
        print("✅ CLIP model loaded")
    return _model, _processor


def embed_image(image: Union[Image.Image, bytes]) -> list:
    """
    Generate a 512-dim embedding from a PIL Image or raw bytes.
    Returns a list of floats (normalized).
    """
    model, processor = _load_model()

    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError("Expected PIL Image or bytes")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    
    # Normalize
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].cpu().numpy().tolist()


def embed_text(text: str) -> list:
    """
    Generate a 512-dim embedding from text using CLIP's text encoder.
    Useful for text-based search over image embeddings.
    """
    model, processor = _load_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].cpu().numpy().tolist()


if __name__ == "__main__":
    # Quick test
    print("Testing CLIP embeddings...")
    vec = embed_text("hydraulic hose leak on excavator boom")
    print(f"Text embedding dim: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")
    
    # Test with a dummy image
    dummy = Image.new("RGB", (224, 224), color=(128, 64, 32))
    vec2 = embed_image(dummy)
    print(f"Image embedding dim: {len(vec2)}")
    print("✅ CLIP embeddings working!")