#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <iostream>
#include <cstdint>

// Declare the kernel
extern "C" 
{
    void launch_cuda_sbvr_mm_T(
        uint32_t* l_bvr,
        uint8_t* l_coeff_idx,
        __half* l_coeff_cache,
        uint32_t* r_bvr,
        uint8_t* r_coeff_idx,
        __half* r_coeff_cache,
        __half* bias,
        __half* out,
        int out_rows,
        int out_cols,
        int l_num_sums,
        int r_num_sums,
        int l_cache_size,
        int r_cache_size,
        int cgroup_per_inner_vec,
        int bvr_per_cgroup);
}

// PyTorch wrapper
torch::Tensor sbvr_mm_T(torch::Tensor l_bvr,
                        torch::Tensor l_coeff_idx,
                        torch::Tensor l_coeff_cache,
                        torch::Tensor r_bvr,
                        torch::Tensor r_coeff_idx,
                        torch::Tensor r_coeff_cache,
                        torch::Tensor bias)
{
    // Tensor shapes:
    // l_bvr: [out_rows, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // l_coeff_idx: [num_cgroups]
    // l_coeff_cache: [cache_size, num_sums]
    // r_bvr: [out_cols, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // r_coeff_idx: [num_cgroups]
    // r_coeff_cache: [cache_size, num_sums]

    int out_rows = l_bvr.size(0);
    int out_cols = r_bvr.size(0);
    int l_num_sums = l_bvr.size(1);
    int r_num_sums = r_bvr.size(1);
    int l_cache_size = l_coeff_cache.size(0);
    int r_cache_size = r_coeff_cache.size(0);
    int cgroup_per_inner_vec = l_bvr.size(2);
    int bvr_per_cgroup = l_bvr.size(3);
    // int num_cgroups = l_coeff_idx.size(0);

    auto out = torch::empty({out_rows, out_cols},
                         torch::dtype(torch::kFloat16).device(l_bvr.device()));

    __half* bias_ptr = nullptr;
    if (bias.size(0) != 1) 
        bias_ptr = reinterpret_cast<__half*>(bias.data_ptr<at::Half>());

    launch_cuda_sbvr_mm_T(
        l_bvr.data_ptr<uint32_t>(),
        l_coeff_idx.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(l_coeff_cache.data_ptr<at::Half>()),
        r_bvr.data_ptr<uint32_t>(),
        r_coeff_idx.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(r_coeff_cache.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        out_rows,
        out_cols,
        l_num_sums,
        r_num_sums,
        l_cache_size,
        r_cache_size,
        cgroup_per_inner_vec,
        bvr_per_cgroup);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sbvr_mm_T", &sbvr_mm_T, 
          "SBVR Matrix-Matrix_Tranposed Multiplication kernel");
}
