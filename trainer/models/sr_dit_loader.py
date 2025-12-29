import torch
from .sr_dit_model import SRDiT
from .base_loader import load_common_components
from diffusers import AutoencoderKL

def load_sr_dit_components(args, device=None, weight_dtype=torch.float32):
    """
    Loads components specifically for SR-DiT model.
    """
    device_map = None
    if device:
        device_map = {"": str(device)}

    # Load common components
    noise_scheduler, tokenizer, text_encoder = load_common_components(args, device, weight_dtype)

    # Load VAE (Default to kaiyuyue/FLUX.2-dev-vae)
    vae_path = args.vae_path or "kaiyuyue/FLUX.2-dev-vae"
    
    # Try AutoencoderKLFlux2 if it exists, otherwise fallback to AutoencoderKL
    try:
        from diffusers import AutoencoderKLFlux2
        VAEClass = AutoencoderKLFlux2
    except ImportError:
        from diffusers import AutoencoderKL
        VAEClass = AutoencoderKL

    vae = VAEClass.from_pretrained(
        vae_path,
        device_map=device_map,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True if device_map else False
    )

    # Init SR-DiT
    print("Initializing SR-DiT from scratch...")
    
    # Latent size calculation based on resolution and VAE scaling (usually 8)
    resolution = getattr(args, "resolution", 256)
    if isinstance(resolution, list): resolution = resolution[0]
    input_size = resolution // 8
    
    config = getattr(args, "model_config", {}) or {} 
    
    transformer = SRDiT(
        input_size=input_size,
        patch_size=config.get("patch_size", 2),
        in_channels=vae.config.latent_channels,
        hidden_size=config.get("hidden_size", 768),
        depth=config.get("depth", 12),
        num_heads=config.get("num_heads", 12),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        context_dim=text_encoder.config.hidden_size,
        cls_token_dim=config.get("cls_token_dim", 768),
        z_dims=config.get("z_dims", [1024]),
        projector_dim=config.get("projector_dim", 2048)
    )
    
    transformer = transformer.to(dtype=torch.float32)

    return noise_scheduler, tokenizer, text_encoder, vae, transformer
