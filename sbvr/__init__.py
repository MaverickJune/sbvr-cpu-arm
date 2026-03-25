from .core import (sbvr, _sbvr_serialized, load, mm_T)
# from sbvr.sbvr_cuda import _sbvr_cuda_init, _sbvr_mm_T, _sbvr_row_deq_mm_T
from sbvr.sbvr_cpu import _sbvr_init_pool, _sbvr_finalize_pool, _sbvr_neon_test, _sbvr_cpu_mm_T
# [v2] Import v2 CPU kernel bindings (CUDA-style layout, optimized for Neoverse-V2)
from sbvr.sbvr_cpu_v2 import _sbvr_v2_init_pool, _sbvr_v2_finalize_pool, _sbvr_cpu_v2_mm_T
import torch, atexit

torch.serialization.add_safe_globals([_sbvr_serialized])
# _sbvr_cuda_init()

_sbvr_init_pool(-1)
atexit.register(_sbvr_finalize_pool)

# [v2] Initialize v2 thread pool
_sbvr_v2_init_pool(-1)
atexit.register(_sbvr_v2_finalize_pool)