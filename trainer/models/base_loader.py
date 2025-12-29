import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_common_components(args, device=None, weight_dtype=torch.float32):
    """
    Loads common components (Scheduler, Tokenizer, Text Encoder, VAE) shared across models.
    """
    device_map = None
    if device:
        device_map = {"": str(device)}

    # Load scheduler
    # Some models might need specific scheduler args, but FlowMatchEulerDiscreteScheduler is common here.
    if args.pretrained_model_name_or_path:
        try:
            noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler",
            )
        except:
             noise_scheduler = FlowMatchEulerDiscreteScheduler()
    else:
        noise_scheduler = FlowMatchEulerDiscreteScheduler()

    # Determine paths
    pretrained_path = args.pretrained_model_name_or_path
    
    # Load Tokenizer
    tokenizer_path = getattr(args, "text_encoder_path", None) or pretrained_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
    )

    # Load Text Encoder
    text_encoder = AutoModelForCausalLM.from_pretrained(
        tokenizer_path,
        device_map=device_map,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True if device_map else False
    )

    # Load VAE
    # Note: Sana uses AutoencoderDC, ZImage/SRDiT use AutoencoderKL.
    # We can default to KL or make it selectable if needed, but for now specific loaders 
    # might need to handle VAE if it differs significantly.
    # However, checking the files:
    # ZImage: AutoencoderKL
    # SRDiT: AutoencoderKL
    # Sana: AutoencoderDC
    # So we can't fully generalize VAE here without an arg or check.
    # Let's return tokenizer, text_encoder, scheduler and let specific loaders handle VAE 
    # OR we pass a VAE class type.
    
    return noise_scheduler, tokenizer, text_encoder
