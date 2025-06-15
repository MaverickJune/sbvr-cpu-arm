from .core import (sbvr, _sbvr_serialized, load, mm_T)
# from sbvr.sbvr_cuda import _sbvr_cuda_init, _sbvr_mm_T, _sbvr_row_deq_mm_T
from sbvr.sbvr_cpu import _sbvr_init_pool, _sbvr_finalize_pool, _sbvr_neon_test, _sbvr_cpu_mm_T
import torch, atexit

torch.serialization.add_safe_globals([_sbvr_serialized])
# _sbvr_cuda_init()

_sbvr_init_pool(-1)
atexit.register(_sbvr_finalize_pool)