# Save this as stylegan3_ada/__init__.py
import sys
import os

# The "Magic" Path Injection
# This allows the files in the /training folder to see 
# the /dnnlib and /torch_utils folders as if they were in the root.
base_path = os.path.dirname(os.path.abspath(__file__))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# Export key components for your robust_unsupervised logic
from . import dnnlib
from . import legacy
from .training import networks_stylegan3 as networks