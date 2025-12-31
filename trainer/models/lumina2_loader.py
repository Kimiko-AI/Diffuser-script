import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Lumina2Transformer2DModel,
)
from transformers import AutoTokenizer, Gemma2Model

def load_lumina2_components(args, device=None, weight_dtype=torch.float32):
    """
    Loads components specifically for Lumina 2 model.
    """
    device_map = None
    if device:
        device_map = {"": str(device)}

    # Load scheduler
    if args.pretrained_model_name_or_path:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler",
        )
    else:
        # Default config if not loading from pretrained (though usually we do)
        noise_scheduler = FlowMatchEulerDiscreteScheduler()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=getattr(args, "revision", None),
    )

    # Load Text Encoder (Gemma 2)
    # Gemma 2 is often loaded in bfloat16 or float16 to save memory
    text_encoder_dtype = weight_dtype
    if weight_dtype == torch.float16 or weight_dtype == torch.bfloat16:
        text_encoder_dtype = weight_dtype # Keep consistent
    
    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        device_map=device_map,
        torch_dtype=text_encoder_dtype,
        revision=getattr(args, "revision", None),
        variant=getattr(args, "variant", None),
    )

    # Load VAE (AutoencoderKL)
    # VAE is usually kept in float32 for stability, but we can respect weight_dtype if forced
    # The provided script keeps VAE in float32.
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        device_map=device_map,
        torch_dtype=torch.float32, 
        revision=getattr(args, "revision", None),
        variant=getattr(args, "variant", None),
    )

    # Load Transformer (Lumina2Transformer2DModel)
    model_config = getattr(args, "model_config", None)
    if model_config:
        print(f"Initializing Lumina2Transformer2DModel from config: {model_config}")
        # Filter config keys
        config = {k: v for k, v in model_config.items() if not k.startswith("_")}
        transformer = Lumina2Transformer2DModel(**config)
    else:
        transformer = Lumina2Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="transformer", 
            device_map=device_map,
            torch_dtype=weight_dtype,
            revision=getattr(args, "revision", None),
            variant=getattr(args, "variant", None),
        )

    return noise_scheduler, tokenizer, text_encoder, vae, transformer
