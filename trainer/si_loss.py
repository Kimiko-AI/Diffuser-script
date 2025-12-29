import torch
import numpy as np
import math

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            cfm_weighting="uniform",
            encoders=[], 
            accelerator=None, 
            apply_time_shift=False,
            shift_base=4096,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.cfm_weighting = cfm_weighting
        self.apply_time_shift = apply_time_shift
        self.shift_base = shift_base

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None, cls_token=None,
                 time_input=None, noises=None,):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if time_input is None:
            if self.weighting == "uniform":
                time_input = torch.rand((images.shape[0], 1, 1, 1))
            elif self.weighting == "lognormal":
                # sample timestep according to log-normal distribution of sigmas following EDM
                rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
                sigma = rnd_normal.exp()
                if self.path_type == "linear":
                    time_input = sigma / (1 + sigma)
                elif self.path_type == "cosine":
                    time_input = 2 / np.pi * torch.atan(sigma)
        
        if self.apply_time_shift:
            shift_dim = images.shape[1] * images.shape[2] * images.shape[3]
            shift = math.sqrt(shift_dim / self.shift_base)
            time_input = (shift * time_input) / (1 + (shift - 1) * time_input)
            time_input = torch.clamp(time_input, 0.0, 1.0)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        if noises is None:
            noises = torch.randn_like(images)
            noises_cls = torch.randn_like(cls_token)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        
        # Adjust dimensions for cls_token broadcasting
        # alpha_t is (B, 1, 1, 1), cls_token is (B, D)
        alpha_t_cls = alpha_t.squeeze(-1).squeeze(-1)
        sigma_t_cls = sigma_t.squeeze(-1).squeeze(-1)
        d_alpha_t_cls = d_alpha_t.squeeze(-1).squeeze(-1) if isinstance(d_alpha_t, torch.Tensor) else d_alpha_t
        d_sigma_t_cls = d_sigma_t.squeeze(-1).squeeze(-1) if isinstance(d_sigma_t, torch.Tensor) else d_sigma_t
        
        cls_input = alpha_t_cls * cls_token + sigma_t_cls * noises_cls
        
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
            cls_target = d_alpha_t_cls * cls_token + d_sigma_t_cls * noises_cls
        else:
            raise NotImplementedError()

        model_output, cls_output = model(model_input, time_input.flatten(), **model_kwargs,
                                                    cls_token=cls_input)

        #denoising_loss
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss_cls = mean_flat((cls_output - cls_target) ** 2)

        cfm_target = torch.roll(model_target, shifts=1, dims=0)
        cfm_target_cls = torch.roll(cls_target, shifts=1, dims=0)
        
        if self.cfm_weighting == "uniform":
            cfm_loss = -((model_output - cfm_target) ** 2).mean()
            cfm_loss_cls = -((cls_output - cfm_target_cls) ** 2).mean()
        elif self.cfm_weighting == "linear":
            cfm_loss = -(((model_output - cfm_target) ** 2) * time_input).mean()
            cfm_loss_cls = -(((cls_output - cfm_target_cls) ** 2) * time_input).mean()

        return denoising_loss, time_input, noises, denoising_loss_cls, cfm_loss, cfm_loss_cls
