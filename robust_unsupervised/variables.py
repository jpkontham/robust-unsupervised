from .prelude import *
class StyleGAN3Hook:
    def __init__(self, G, styles_dict):
        self.styles = styles_dict
        self.hooks = []
        
        # We iterate over synthesis modules because that's where 
        # the modulated convolutions live.
        for name, module in G.synthesis.named_modules():
            # In SG3, both SynthesisInput and SynthesisLayer have an 'affine' 
            # layer that transforms W into the style vector.
            if hasattr(module, 'affine'):
                # Normalize name to match your SVariable dictionary keys
                m_name = name.replace('.', '_')
                
                if m_name in self.styles:
                    # We hook the .affine module specifically.
                    # This intercepts the style right before it enters modulated_conv2d.
                    h = module.affine.register_forward_hook(self._make_hook(m_name))
                    self.hooks.append(h)

    def _make_hook(self, name):
        # We use a closure to lock in the 'name' for each specific layer
        def hook(module, input, output):
            # We override the output of the affine layer with our 
            # optimized StyleSpace parameter.
            return self.styles[name]
        return hook

    def remove(self):
        # Crucial for Kaggle: prevents memory leaks and hook-clash 
        # between different images in your benchmark.
        for h in self.hooks:
            h.remove()
        self.hooks = []

class Variable(nn.Module):
    def __init__(self, G: networks.Generator, data: torch.Tensor):
        super().__init__()
        self.G = G
        self.data = data

    # ------------------------------------
    
    @staticmethod
    def sample_from(G: networks.Generator, batch_size: int = 1):
        raise NotImplementedError

    @staticmethod
    def sample_random_from(G: networks.Generator, batch_size: int = 1):
        raise NotImplementedError

    def to_input_tensor(self):
        raise NotImplementedError

    # ------------------------------------

    def parameters(self):
        return [self.data]

    def to_image(self):
        return self.render_image(self.to_input_tensor())

    def render_image(self, ws_or_styles): 
        if isinstance(ws_or_styles, (dict, nn.ParameterDict)):
            hooks = StyleGAN3Hook(self.G, ws_or_styles)
            # Dummy WS is ignored by the hooks
            batch_size = next(iter(ws_or_styles.values())).shape[0]
            dummy_ws = torch.zeros(batch_size, self.G.num_ws, self.G.w_dim).to(next(self.G.parameters()).device)
            img = self.G.synthesis(dummy_ws, force_fp32=True)
            hooks.remove()
        else:
            img = self.G.synthesis(ws_or_styles, force_fp32=True)
        
        return (img + 1.0) / 2.0

    def detach(self):
        data = self.data.detach().requires_grad_(self.data.requires_grad)
        data = nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data
        return self.__class__(self.G, data)

    def clone(self):
        data = self.data.detach().clone().requires_grad_(self.data.requires_grad)
        data = nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data
        return self.__class__(self.G, data)

    def interpolate(self, other: "Variable", alpha: float = 0.5):
        assert self.G == other.G
        return self.__class__(self.G, self.data.lerp(other.data, alpha))

    def __add__(self, other: "Variable"):
        return self.from_data(self.data + other.data)

    def __sub__(self, other: "Variable"):
        return self.from_data(self.data - other.data)

    def __mul__(self, scalar: float):
        return self.from_data(self.data * scalar)

    def unbind(self):
        """
        Splits this (batched) variable into a a list of variables with batch size 1. 
        """
        return [
            self.__class__(
                self.G,
                nn.Parameter(p.unsqueeze(0))
                if isinstance(self.data, nn.Parameter)
                else p.unsqueeze(0),
            )
            for p in self.data
        ]
    

class WVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = G.mapping.w_avg.reshape(1, G.w_dim).repeat(batch_size, 1)

        return WVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = G.mapping(
            torch.randn(batch_size, G.z_dim).cuda(),
            None,
            skip_w_avg_update=True,
        )[:, 0]

        return WVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        return self.data.unsqueeze(1).repeat(1, self.G.num_ws, 1)

    @torch.no_grad()
    def truncate(self, truncation: float=1.0):
        assert 0.0 <= truncation <= 1.0
        # CHANGE: Use self.G.w_dim instead of 512
        self.data.lerp_(self.G.mapping.w_avg.reshape(1, self.G.w_dim), 1.0 - truncation)
        return self


class WpVariable(Variable):
    def __init__(self, G, data: torch.Tensor):
        super().__init__(G, data)

    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = WVariable.to_input_tensor(WVariable.sample_from(G, batch_size))

        return WpVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        # CHANGE: Simplified for StyleGAN3
        z = torch.randn(batch_size, G.z_dim).cuda()
        # StyleGAN3 mapping network returns [batch, num_ws, w_dim] automatically
        data = G.mapping(z, None, skip_w_avg_update=True)
        
        return WpVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        return self.data

    def mix(self, other: "WpVariable", num_layers: float):
        return WpVariable(
            self.G,
            torch.cat(
                (self.data[:, :num_layers, :], other.data[:, num_layers:, :]), dim=1
            ),
        )

    @staticmethod
    def from_W(W: WVariable):
        return WpVariable(
            W.G, nn.parameter.Parameter(W.to_input_tensor())
        )

    @torch.no_grad()
    def truncate(self, truncation=1.0, *, layer_start = 0, layer_end: Optional[int] = None):
        assert 0.0 <= truncation <= 1.0
        mu = self.G.mapping.w_avg
        # CHANGE: Use self.G.w_dim instead of 512
        target = mu.reshape(1, 1, self.G.w_dim).repeat(1, self.G.mapping.num_ws, 1)
        self.data[:, layer_start:layer_end].lerp_(target[:, layer_start:layer_end], 1.0 - truncation)
        return self


class WppVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        # Correct: Uses G.w_dim
        data = WVariable.sample_from(G, batch_size).to_input_tensor().repeat(1, G.w_dim, 1)
        return WppVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        # Correct: Uses G.w_dim
        data = (
            WVariable.sample_random_from(G, batch_size)
            .to_input_tensor()
            .repeat(1, G.w_dim, 1)
        )
        return WppVariable(G, nn.Parameter(data))

    @staticmethod # Fixed: Removed extra space/indentation
    def from_w(W: WVariable):
        # Correct: Dynamically calculated repeat count
        data = W.data.detach().repeat(1, W.G.w_dim * W.G.num_ws, 1)
        return WppVariable(W.G, nn.parameter.Parameter(data))

    @staticmethod # Fixed: Removed double decorator
    @torch.no_grad()
    def from_Wp(Wp: WpVariable):
    # Ensure W++ expansion is mathematically aligned with SG3 channels
    # Wpp usually expands W+ [1, 16, 512] -> [1, ~9000, 512]
    # We use repeat_interleave based on the model's actual dimensionality
        data = Wp.data.detach().repeat_interleave(Wp.G.w_dim, dim=1)
        return WppVariable(Wp.G, nn.parameter.Parameter(data))

    def to_input_tensor(self):
        return self.data
        
class SVariable(Variable):
    def __init__(self, G, styles_dict: dict):
        super().__init__(G, nn.ParameterDict({
            k.replace('.', '_'): nn.Parameter(v) 
            for k, v in styles_dict.items()
        }))

    @staticmethod
    @torch.no_grad()
    def from_Wp(Wp: WpVariable):
        G, ws, styles = Wp.G, Wp.to_input_tensor(), {}
        for name, module in G.synthesis.named_modules():
            if hasattr(module, 'affine'):
            # ROBUST FIX: Extracting the numeric index more safely
            # Splitting by '_' and filtering for the numeric part (e.g., 'L0' -> 0)
               try:
                # Most SG3 models use 'L0_...' format
                   layer_idx = int(name.split('_')[0][1:]) 
               except (ValueError, IndexError):
                # Fallback for models with different naming schemes
                   import re
                   layer_idx = int(re.search(r'\d+', name).group())
                
               styles[name] = module.affine(ws[:, layer_idx]).detach()
        return SVariable(G, styles)

    def to_input_tensor(self):
        return self.data
