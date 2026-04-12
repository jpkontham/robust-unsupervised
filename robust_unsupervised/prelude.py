from typing import *
import copy
import os
import functools
import sys
import torch.optim as optim
import tqdm
import dataclasses
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import shutil
from functools import partial
import itertools
import warnings
from warnings import warn
import datetime
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid

# --- THE CHANGE IS HERE ---
# Instead of 'import dnnlib', we import from your new subfolder
import stylegan3_ada.dnnlib as dnnlib
import stylegan3_ada.legacy as legacy
import stylegan3_ada.training.networks_stylegan3 as networks
# ---------------------------

from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
from dataclasses import dataclass, field

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Standard Global for your 5-day sprint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", r"Named tensors and all their associated APIs.*")
warnings.filterwarnings("ignore", r"Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", r"The parameter 'pretrained' is deprecated.*")
