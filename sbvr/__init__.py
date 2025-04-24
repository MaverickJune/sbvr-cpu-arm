from .core import (sbvr, _sbvr_serialized, load, mm_T)
from sbvr.sbvr_cuda import sbvr_cuda_init
import torch

torch.serialization.add_safe_globals([_sbvr_serialized])
sbvr_cuda_init()