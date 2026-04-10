from robust_unsupervised.prelude import *
from robust_unsupervised.variables import *

import shutil
import torch_utils as torch_utils
import torch_utils.misc as misc
import contextlib

import PIL.Image as Image


def open_models(pkl_path: str, float=True, ema=True):
    print(f"Loading StyleGAN3 models from {pkl_path}...")

    with dnnlib.util.open_url(pkl_path) as fp:
        # StyleGAN .pkl files are dictionaries containing G, D, and G_ema
        data = legacy.load_network_pkl(fp)
        G = data['G_ema' if ema else 'G'].cuda().eval()
        D = data['D'].cuda().eval()  # This is the "Art Critic" we need for pFID

        if float:
            G = G.float()
            D = D.float()
            
    # Strictly freeze weights; we only optimize the latent space 'w'
    for param in G.parameters():
        param.requires_grad = False
    for param in D.parameters():
        param.requires_grad = False

    return G, D


def open_image(path: str, resolution: int):
    image = TF.to_tensor(Image.open(path)).cuda().unsqueeze(0)[:, :3]
    image = TF.center_crop(image, min(image.shape[2:]))
    return F.interpolate(image, resolution, mode="area")


def resize_for_logging(x: torch.Tensor, resolution: int) -> torch.Tensor:
    return F.interpolate(
        x,
        size=(resolution, resolution),
        mode="nearest" if x.shape[-1] <= resolution else "area",
    )


@contextlib.contextmanager
def directory(dir_path: str) -> None: 
    "Context manager for entering a directory, while automatically creating it if it does not exist."
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cwd = os.getcwd()
    os.chdir(dir_path)
    yield
    os.chdir(cwd)
