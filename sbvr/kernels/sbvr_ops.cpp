#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <iostream>
#include <cstdint>
#include <assert.h>

// Declare the kernel launcher (templated in the actual .cu file)
void launch_cuda_sbvr_mm_T(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx,__half* r_coeff_cache,
    __half* bias,
    __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size,
    int device_id = 0);

int device_count;
cudaDeviceProp cuda_prop_list[16];

// PyTorch wrapper
torch::Tensor sbvr_mm_T(
                torch::Tensor l_bvr,
                torch::Tensor l_coeff_idx,
                torch::Tensor l_coeff_cache,
                torch::Tensor r_bvr,
                torch::Tensor r_coeff_idx,
                torch::Tensor r_coeff_cache,
                torch::Tensor bias
            )
{
    int M = l_bvr.size(2);
    int N = r_bvr.size(2);
    int K = l_bvr.size(1);
    assert (l_bvr.size(1) == r_bvr.size(1));
    int l_num_sums = l_bvr.size(0) * l_bvr.size(3);
    int r_num_sums = r_bvr.size(0) * r_bvr.size(3);
    int l_cache_size = l_coeff_cache.size(0);
    int r_cache_size = r_coeff_cache.size(0);

    auto out = torch::empty({M, N},
                         torch::dtype(torch::kFloat16).device(l_bvr.device()));
    __half* bias_ptr = nullptr;
    if (bias.size(0) == N)
        bias_ptr = reinterpret_cast<__half*>(bias.data_ptr<at::Half>());
    

    // Call the dispatch kernel
    launch_cuda_sbvr_mm_T(
        l_bvr.data_ptr<uint32_t>(),
        l_coeff_idx.data_ptr(),  // dispatch will cast correctly
        reinterpret_cast<__half*>(l_coeff_cache.data_ptr<at::Half>()),
        r_bvr.data_ptr<uint32_t>(),
        r_coeff_idx.data_ptr(),
        reinterpret_cast<__half*>(r_coeff_cache.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        M, N, K,
        l_num_sums, r_num_sums,
        l_cache_size, r_cache_size);

    return out;
}

void sbvr_cuda_init() 
{
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) 
    {
        std::cerr << "\033[91mSBVR Init:\033[0m No CUDA devices found: " << 
            cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "\033[92mSBVR Init:\033[0m Found " << device_count 
              << " CUDA device(s)." << std::endl;
    for (int device_id = 0; device_id < device_count; ++device_id) 
    {
        err = cudaGetDeviceProperties(&cuda_prop_list[device_id], device_id);
        auto prop = cuda_prop_list[device_id];
        if (err != cudaSuccess) 
        {
            std::cerr << "\tFailed to get properties for device " << device_id 
                      << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }
        std::cout << "\tDevice " << device_id << ": " << prop.name 
                  << " (Compute Capability: " << prop.major << "." 
                  << prop.minor << ")" << std::endl;
    }
    std::cout << "\033[92mSBVR Init:\033[0m" 
              << " CUDA Initialization complete." << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_sbvr_cuda_init", &sbvr_cuda_init,
        "Initialization function for the SBVR CUDA kernels");
    m.def("_sbvr_mm_T", &sbvr_mm_T, 
          py::arg("l_bvr"),
          py::arg("l_coeff_idx"),
          py::arg("l_coeff_cache"),
          py::arg("r_bvr"),
          py::arg("r_coeff_idx"),
          py::arg("r_coeff_cache"),
          py::arg("bias"),
          "SBVR Matrix-Matrix_Transposed Multiplication kernel");
}
