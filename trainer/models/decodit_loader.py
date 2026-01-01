import torch
from .decodit_model import DeCoDiT
from .base_loader import load_common_components

def load_decodit_components(args, device=None, weight_dtype=torch.float32):
    """
    Loads components specifically for DecoDiT model.
    """
    device_map = None
    if device:
        device_map = {"": str(device)}

    # Load common components
    noise_scheduler, tokenizer, text_encoder = load_common_components(args, device, weight_dtype)

    # Init DecoDiT
    print("Initializing DecoDiT from scratch...")
    
    # Pixel space: resolution is the input size (or resolution // patch_size depending on how model treats it)
    # The model takes patch_size in init. input_size usually refers to the spatial dimension of tokens?
    # In the model: x_patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
    # N = x_patches.shape[1]
    # N is approx (H/P * W/P).
    # The 'input_size' arg in SRDiT was used for... wait, looking at decodit_model.py again.
    # It does NOT take 'input_size' in __init__.
    # SRDiT took it? Let me check SRDiT init signature from previous read...
    # Ah, the previous read of `sr_dit_loader.py` showed:
    # transformer = SRDiT(input_size=input_size, ...)
    # But `decodit_model.py` (which was `sr_dit_model.py`) class DeCoDiT __init__ signature I read:
    # def __init__(self, in_channels=4, hidden_size=1152, decoder_hidden_size=64, num_encoder_blocks=18, ...
    # It does NOT have input_size.
    # The previous loader was passing `input_size` but the model definition I read didn't seem to have it.
    # Wait, did I misread the model definition or the loader?
    # Loader: transformer = SRDiT(input_size=input_size, ...)
    # Model: class DeCoDiT(nn.Module): def __init__(self, in_channels=..., ...)
    # The model definition I read DOES NOT have input_size.
    # This implies the user might have updated `sr_dit_model.py` (to DeCoDiT) but not the loader.
    # So I should remove `input_size` from instantiation in the loader.
    
    resolution = getattr(args, "resolution", 256)
    if isinstance(resolution, list): resolution = resolution[0]
    
    config = getattr(args, "model_config", {}) or {} 
    
    transformer = DeCoDiT(
        in_channels=3, # Pixel space RGB
        hidden_size=config.get("hidden_size", 1152),
        decoder_hidden_size=config.get("decoder_hidden_size", 64),
        num_encoder_blocks=config.get("depth", 28), # Mapping 'depth' to 'num_encoder_blocks' if that was the intent, or just check keys
        num_decoder_blocks=config.get("num_decoder_blocks", 4),
        num_text_blocks=config.get("num_text_blocks", 4),
        patch_size=config.get("patch_size", 2),
        txt_embed_dim=text_encoder.config.hidden_size,
        txt_max_length=config.get("txt_max_length", 100), # Maybe derived from tokenizer?
        num_heads=config.get("num_heads", 16)
    )
    
    transformer = transformer.to(dtype=torch.float32)

    return noise_scheduler, tokenizer, text_encoder, None, transformer