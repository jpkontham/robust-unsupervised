import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import *
from .prelude import *
from lpips import LPIPS

class MultiscaleLPIPS(nn.Module):
    def __init__(
        self,
        discriminator: Any = None,  
        min_loss_res: int = 16,
        level_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        l1_weight: float = 0.1,
        adv_weight: float = 0.005 
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.l1_weight = l1_weight
        self.adv_weight = adv_weight
        
        # Initialize LPIPS and move to GPU
        self.lpips_network = LPIPS(net="vgg", verbose=False).cuda().eval()
        self.D = discriminator.cuda().eval() if discriminator is not None else None

    def measure_lpips(self, x, y, mask):
        if mask is not None:
            noise = (torch.randn_like(x).to(x.device) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)
        return self.lpips_network(x, y, normalize=True).mean() 

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None, use_adv: bool = False):
        x = f_hat(x_clean) 

        loss_adv = torch.tensor(0.0).to(x.device)
        if use_adv and self.D is not None:
            x_norm = (x * 2.0) - 1.0
            logits = self.D(x_norm, None)
            loss_adv = torch.nn.functional.softplus(-logits).mean() 

        losses = []
        curr_x, curr_y, curr_mask = x, y, mask

        for weight in self.weights:
            if curr_y.shape[-1] <= self.min_loss_res:
                break
        
            if weight > 0:
                loss = self.measure_lpips(curr_x, curr_y, curr_mask)
                losses.append(weight * loss)

            new_size = curr_y.shape[-1] // 2
            curr_x = F.interpolate(curr_x, size=(new_size, new_size), mode='area')
            curr_y = F.interpolate(curr_y, size=(new_size, new_size), mode='area')
            if curr_mask is not None:
                curr_mask = F.interpolate(curr_mask, size=(new_size, new_size), mode='area')

        total_lpips = torch.stack(losses).sum(dim=0) if len(losses) > 0 else torch.tensor(0.0).to(x.device)
        l1 = self.l1_weight * F.l1_loss(curr_x, curr_y)

        return total_lpips + l1 + (self.adv_weight * loss_adv if use_adv else 0.0)
