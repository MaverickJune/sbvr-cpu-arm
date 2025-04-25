#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>
#include <cassert>

cudaDeviceProp cuda_prop_list[16];
int device_count = 0;

// CUDA kernel launcher
void launch_cuda_sbvr_mm_T(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size,
    int device_id = 0);

// Entry point (takes raw addresses)
void sbvr_mm_T(
    uintptr_t l_bvr_ptr,
    uintptr_t l_coeff_idx_ptr,
    uintptr_t l_coeff_cache_ptr,
    uintptr_t r_bvr_ptr,
    uintptr_t r_coeff_idx_ptr,
    uintptr_t r_coeff_cache_ptr,
    uintptr_t bias_ptr,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size,
    int device_id = 0
) {
    // Allocate output tensor
    // Set device
    // cudaSetDevice(device_id);
    // auto out = torch::empty({M, N}, 
    //     torch::dtype(torch::kFloat16).device(torch::kCUDA));

    // Dispatch kernel
    // launch_cuda_sbvr_mm_T(
    //     reinterpret_cast<uint32_t*>(l_bvr_ptr),
    //     reinterpret_cast<void*>(l_coeff_idx_ptr),
    //     reinterpret_cast<__half*>(l_coeff_cache_ptr),
    //     reinterpret_cast<uint32_t*>(r_bvr_ptr),
    //     reinterpret_cast<void*>(r_coeff_idx_ptr),
    //     reinterpret_cast<__half*>(r_coeff_cache_ptr),
    //     (bias_ptr == 0) ? nullptr : reinterpret_cast<__half*>(bias_ptr),
    //     reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
    //     M, N, K,
    //     l_num_sums, r_num_sums,
    //     l_cache_size, r_cache_size,
    //     device_id
    // );
    return;
    // return out;
}

// Optional init
void sbvr_cuda_init() {
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) 
    {
        std::cerr << "SBVR Init) No CUDA devices found: " << 
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

// Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sbvr_cuda_init", &sbvr_cuda_init, "Init function for CUDA kernels");
    m.def("sbvr_mm_T", &sbvr_mm_T,
        py::arg("l_bvr_ptr"),
        py::arg("l_coeff_idx_ptr"),
        py::arg("l_coeff_cache_ptr"),
        py::arg("r_bvr_ptr"),
        py::arg("r_coeff_idx_ptr"),
        py::arg("r_coeff_cache_ptr"),
        py::arg("bias_ptr") = 0,
        py::arg("M"),
        py::arg("N"),
        py::arg("K"),
        py::arg("l_num_sums"),
        py::arg("r_num_sums"),
        py::arg("l_cache_size"),
        py::arg("r_cache_size"),
        py::arg("device_id") = 0,
        "SBVR matmul with raw pointers"
    );
}
