import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Lumina2Transformer2DModel,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        # Default config
        noise_scheduler = FlowMatchEulerDiscreteScheduler()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=getattr(args, "revision", None),
    )

    # Load Text Encoder (Gemma 2)
    text_encoder_dtype = weight_dtype
    if weight_dtype == torch.float16 or weight_dtype == torch.bfloat16:
        text_encoder_dtype = weight_dtype 
    
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder",
        device_map=device_map,
        torch_dtype=text_encoder_dtype,
        revision=getattr(args, "revision", None),
        variant=getattr(args, "variant", None),
    )

    # Load VAE (AutoencoderKL)
    vae = AutoencoderKL.from_pretrained(
        args.vae_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        revision=getattr(args, "revision", None),
        variant=getattr(args, "variant", None),
    )

    # Load Transformer
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
            torch_dtype=torch.float32,
            revision=getattr(args, "revision", None),
            variant=getattr(args, "variant", None),
        )

    return noise_scheduler, tokenizer, text_encoder, vae, transformer
