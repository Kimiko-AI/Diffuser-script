from .zimage_loader import load_zimage_components
from .sana_loader import load_sana_components
from .sr_dit_loader import load_sr_dit_components
from .lumina2_loader import load_lumina2_components
from .zimage_wrapper import ZImageWrapper
from .sana_wrapper import SanaWrapper
from .sr_dit_wrapper import SRDiTWrapper
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
    elif model_type == "sr_dit":
        return load_sr_dit_components(args, device=device, weight_dtype=weight_dtype)
    elif model_type == "lumina2":
        return load_lumina2_components(args, device=device, weight_dtype=weight_dtype)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def get_model_wrapper(model_type, **kwargs):
    if model_type == "sana":
        return SanaWrapper(**kwargs)
    elif model_type == "zimage":
        return ZImageWrapper(**kwargs)
    elif model_type == "sr_dit":
        return SRDiTWrapper(**kwargs)
    elif model_type == "lumina2":
        return Lumina2Wrapper(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
