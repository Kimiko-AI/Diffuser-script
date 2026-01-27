from .zimage_loader import load_zimage_components
from .lumina2_loader import load_lumina2_components
from .zimage_wrapper import ZImageWrapper
from .lumina2_wrapper import Flux2TrainingWrapper
import torch

def load_models(args, device=None, weight_dtype=torch.float32):
    """
    Factory function to load model components based on model_type.
    """
    model_type = getattr(args, "model_type", "zimage")
    
    if model_type == "zimage":
        return load_zimage_components(args, device=device, weight_dtype=weight_dtype)
    elif model_type == "lumina2":
        return load_lumina2_components(args, device=device, weight_dtype=weight_dtype)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: zimage, lumina2")

def get_model_wrapper(model_type, **kwargs):
    if model_type == "zimage":
        return ZImageWrapper(**kwargs)
    elif model_type == "lumina2":
        return Flux2TrainingWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: zimage, lumina2")