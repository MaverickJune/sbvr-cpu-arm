from .core import sbvr, load_sbvr, save_sbvr
import torch

torch.serialization.add_safe_globals([sbvr])