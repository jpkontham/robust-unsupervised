import torch 


class NGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                # Ensure the parameter itself hasn't corrupted
                assert param.isnan().sum().item() == 0, "Parameter contains NaNs"
                
                # Skip if no gradient exists (common in conditional loops)
                if param.grad is None:
                    continue
                
                g = param.grad
                g = g / (g.norm(dim=-1, keepdim=True) + 1e-8)
                g = torch.nan_to_num(
                    g, nan=0.0, posinf=0.0, neginf=0.0
                )
                
                # Apply update based on the group's learning rate
                param -= group["lr"] * g
