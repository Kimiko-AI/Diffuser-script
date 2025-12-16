
import torch
import torch.nn as nn
from trainer.ZImage import ZImageWrapper

class MockConfig:
    def __init__(self):
        self.latent_channels = 4
        self.shift_factor = 0.0
        self.scaling_factor = 1.0

class MockVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
    
    def encode(self, x):
        return self # Mock
    
    @property
    def latent_dist(self):
        return self
        
    def sample(self):
        return torch.randn(1, 4, 32, 32)

class MockTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {}
    def forward(self, *args, **kwargs):
        # Return list of outputs as per ZImageWrapper expectation
        return [torch.randn(1, 4, 32, 32).unsqueeze(2)] # (B, C, T, H, W)

def test_wrapper():
    transformer = MockTransformer()
    vae = MockVAE()
    text_encoder = nn.Linear(10, 10) # Mock
    tokenizer = None
    noise_scheduler = None
    
    wrapper = ZImageWrapper(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler
    )
    
    print("Wrapper initialized.")
    print(f"Refiner present: {hasattr(wrapper, 'refiner')}")
    if hasattr(wrapper, 'refiner'):
        print(f"Refiner: {wrapper.refiner}")
        params = list(wrapper.parameters())
        print(f"Total trainable params groups: {len(params)}")
        # Check if refiner params are in wrapper.parameters()
        refiner_params = list(wrapper.refiner.parameters())
        transformer_params = list(wrapper.transformer.parameters())
        
        print(f"Transformer params: {len(transformer_params)}")
        print(f"Refiner params: {len(refiner_params)}")
        print(f"Wrapper params: {len(params)}")
        
        assert len(params) == len(transformer_params) + len(refiner_params)
        print("Parameter count check passed.")

if __name__ == "__main__":
    test_wrapper()
