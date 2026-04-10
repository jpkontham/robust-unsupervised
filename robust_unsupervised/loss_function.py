from .prelude import *
from lpips import LPIPS

class MultiscaleLPIPS:
    def __init__(
        self,
        discriminator: Any,  # NEW: Pass the loaded Discriminator here
        min_loss_res: int = 16,
        level_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        l1_weight: float = 0.1,
        adv_weight: float = 0.005 # NEW: Controls how strictly to enforce realism
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.l1_weight = l1_weight
        self.adv_weight = adv_weight
        self.lpips_network = LPIPS(net="vgg", verbose=False).cuda()
        self.D = discriminator.cuda().eval() # NEW: Store the Discriminator

    def measure_lpips(self, x, y, mask):
        if mask is not None:
            # To avoid biasing the results towards black pixels, but random noise in the masked areas
            noise = (torch.randn_like(x) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)

        return self.lpips_network(x, y, normalize=True).mean() 

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None):
        x = f_hat(x_clean)

        losses = []

        # --- NEW: Adversarial Loss (The pFID Fix) ---
        # Calculate D's score on the high-resolution generated image (x)
        # Minimizing -D(x) forces the generator to produce realistic textures
        loss_adv = -self.D(x, None).mean() 

        # Create temporary variables for the downsampling loop 
        # so we don't overwrite the original tensors.
        curr_x = x
        curr_y = y
        curr_mask = mask

        if curr_mask is not None:
            curr_mask = F.interpolate(curr_mask, curr_y.shape[-1], mode="area")

        for weight in self.weights:
            # At extremely low resolutions, LPIPS stops making sense, so omit those
            if curr_y.shape[-1] <= self.min_loss_res:
                break
            
            if weight > 0:
                loss = self.measure_lpips(curr_x, curr_y, curr_mask)
                losses.append(weight * loss)

            if curr_mask is not None:
                curr_mask = F.avg_pool2d(curr_mask, 2)

            curr_x = F.avg_pool2d(curr_x, 2)
            curr_y = F.avg_pool2d(curr_y, 2)
        
        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        
        # L1 is computed on the lowest resolution (curr_x, curr_y)
        l1 = self.l1_weight * F.l1_loss(curr_x, curr_y)

        # --- NEW: Return Fidelity + Realism ---
        return total + l1 + (self.adv_weight * loss_adv)
