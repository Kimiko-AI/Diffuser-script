import torch
import torch.nn as nn
import timm
from torchvision import transforms

class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_large_patch14_reg4_dinov2.lvd142m', device='cuda'):
        super().__init__()
        self.device = device
        # Initialize model
        # Using dynamic_img_size=True as per reference
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.config = timm.data.resolve_model_data_config(self.model)
        self.normalize = transforms.Normalize(mean=self.config['mean'], std=self.config['std'])

    @torch.no_grad()
    def extract_features(self, images):
        """
        images: (B, C, H, W) tensor, normalized to [0, 1] or similar
        Returns: 
            cls_token: (B, Dim)
            patch_tokens: (B, N_patches, Dim)
        """
        B, C, H, W = images.shape
        
        # 1. Resize if needed to match patch alignment
        # Usually input to this pipeline is already sized well (e.g. 256 or 512)
        # But DINO expects specific patch multiples.
        # Assuming images are already tensors.
        
        # We need to apply DINO specific normalization.
        # Images coming in are usually [-1, 1] from VAE pipeline or [0, 1].
        # User confirmed input is [-1, 1].
        # Convert to [0, 1] first
        x = images * 0.5 + 0.5
        x = self.normalize(x)
        
        # 2. Forward
        features = self.model.forward_features(x)
        
        # 3. Extract
        n_prefix = self.model.num_prefix_tokens
        cls_token = features[:, 0] # First token is usually CLS (check model specific)
        # Note: Reg4 models have registers. 
        # Structure: [CLS, Reg1, Reg2, Reg3, Reg4, Patch1, ...]
        # CLS is index 0. Registers are 1..n_prefix-1. Patches are n_prefix..
        
        # Reference code used features[0, n_prefix:] for patches.
        # So cls is likely 0.
        
        patch_tokens = features[:, n_prefix:]
        
        return cls_token, patch_tokens
