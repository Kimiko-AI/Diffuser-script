from .zimage_loader import load_zimage_components
from .sana_loader import load_sana_components
from .decodit_loader import load_decodit_components
from .lumina2_loader import load_lumina2_components
from .zimage_wrapper import ZImageWrapper
from .sana_wrapper import SanaWrapper
from .decodit_wrapper import DecoDiTWrapper
from .lumina2_wrapper import Lumina2Wrapper
import torch

def load_models(args, device=None, weight_dtype=torch.float32):
    """
    Factory function to load model components based on model_type.
    """
    model_type = getattr(args, "model_type", "zimage")
    
    if model_type == "sana":
        return load_sana_components(args, device=device, weight_dtype=weight_dtype)
    elif model_type == "zimage":
        return load_zimage_components(args, device=device, weight_dtype=weight_dtype)
    elif model_type == "decodit":
        return load_decodit_components(args, device=device, weight_dtype=weight_dtype)
    elif model_type == "lumina2":
        return load_lumina2_components(args, device=device, weight_dtype=weight_dtype)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def get_model_wrapper(model_type, **kwargs):
    if model_type == "sana":
        return SanaWrapper(**kwargs)
    elif model_type == "zimage":
        return ZImageWrapper(**kwargs)
    elif model_type == "decodit":
        return DecoDiTWrapper(**kwargs)
    elif model_type == "lumina2":
        return Lumina2Wrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")