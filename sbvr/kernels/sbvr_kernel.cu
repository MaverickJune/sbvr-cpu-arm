
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>

#define K_PER_BVR 4

template <typename LIndexT, typename RIndexT>
__global__ void cuda_naive_sbvr_mm_T(
    uint32_t* l_bvr, LIndexT* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums)
{
    // Tensor shapes:
    // l_bvr: [self.num_sums, K, M]
    // l_coeff_idx: [num_bvr]
    // l_coeff_cache: [cache_size, num_sums]
    // r_bvr: [self.num_sums, K, N]
    // r_coeff_idx: [num_bvr]
    // r_coeff_cache: [cache_size, num_sums]

    float coeff_mult[10][10];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = M * N;
    const int bvr_per_K = K / K_PER_BVR;

    for (int i = tid; i < total_outputs; i += blockDim.x * gridDim.x) 
    {
        const int m = i / N;
        const int n = i % N;
        
        float sum = (bias != nullptr) ? __half2float(bias[n]) : 0.0f;

        // if (tid == 0)
        //     printf("Tid %d (%d, %d), bias: %f, M: %d, N: %d, "
        //         "l_num_sums: %d, r_num_sums: %d, bvr_per_K: %d\n",
        //         tid, m, n, sum, M, N, 
        //         l_num_sums, r_num_sums, bvr_per_K);

        for (int bvr_idx = 0; bvr_idx < bvr_per_K; bvr_idx++)
        {
            int l_coeff_cache_idx = l_coeff_idx[m * bvr_per_K + bvr_idx];
            int r_coeff_cache_idx = r_coeff_idx[n * bvr_per_K + bvr_idx];
            __half* l_coeff_ptr = 
                            &l_coeff_cache[l_coeff_cache_idx * l_num_sums];
            __half* r_coeff_ptr = 
                            &r_coeff_cache[r_coeff_cache_idx * r_num_sums];

            // Precompute the coefficient multiplications
            for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
            {
                const float l_coeff = __half2float(l_coeff_ptr[l_idx]);
                for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                {
                    coeff_mult[l_idx][r_idx] = 
                        l_coeff * __half2float(r_coeff_ptr[r_idx]);
                    // if (tid == 0)
                    //     printf("bvr_idx: %d, l_coeff: %f, "
                    //         "r_coeff: %f, coeff_mult: %f\n", 
                    //         bvr_idx, l_coeff, __half2float(r_coeff_ptr[r_idx]), 
                    //         l_coeff * __half2float(r_coeff_ptr[r_idx]));
                }
            }

            for (int k = 0; k < K_PER_BVR; k++)
            {
                const int k_idx = bvr_idx * K_PER_BVR + k;   
                for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
                {
                    // l_bvr: [self.num_sums, K, M]
                    const uint32_t l = l_bvr[l_idx * K * M + k_idx * M + m];
                    for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                    {
                        // r_bvr: [self.num_sums, K, N]
                        const uint32_t r = r_bvr[r_idx * K * N + k_idx * N + n];
                        const uint32_t lr = l & r;
                        const float lr_popc = (float)__popc(lr);
                        sum += lr_popc * coeff_mult[l_idx][r_idx];
                        // if (tid == 0)
                        //     printf("bvr_idx: %d, l: %u, r: %u, lr: %u, "
                        //         "lr_popc: %f, coeff_mult: %f, sum: %f\n", 
                        //         bvr_idx, l, r, lr, lr_popc, 
                        //         coeff_mult[l_idx][r_idx], sum);
                    }
                }
            }

        }

        // Store the result in the output matrix
        out[i] = __float2half(sum);
    }
    
}

template <typename LIndexT, typename RIndexT, int L_NUM_SUMS, int R_NUM_SUMS>
__global__ void cuda_8x4_sbvr_mm_T(
    uint32_t* l_bvr, LIndexT* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K)
{
    // Tensor shapes:
    // l_bvr: [self.num_sums, K, M]
    // l_coeff_idx: [num_bvr]
    // l_coeff_cache: [cache_size, L_NUM_SUMS]
    // r_bvr: [self.num_sums, K, N]
    // r_coeff_idx: [num_bvr]
    // r_coeff_cache: [cache_size, R_NUM_SUMS]

    __half coeff_mult[L_NUM_SUMS][R_NUM_SUMS];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = M * N;
    const int bvr_per_K = K / K_PER_BVR;

    if (tid == 0)
        printf("Tid %d, M: %d, N: %d, "
            "l_num_sums: %d, r_num_sums: %d, bvr_per_K: %d\n",
            tid, M, N, 
            L_NUM_SUMS, R_NUM_SUMS, bvr_per_K);
    
}

template <typename LIndexT, typename RIndexT>
void launch_naive_sbvr_kernel(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount * 8;
    dim3 threads = 32;

    std::cout << "Launching naive SBVR kernel <" 
              << typeid(LIndexT).name() << ", " 
              << typeid(RIndexT).name() << ", "
              << "l_num_sums: " << l_num_sums << ", "
              << "r_num_sums: " << r_num_sums << ", "
              << "M: " << M << ", "
              << "N: " << N << ", "
              << "K: " << K << ", "
              << "blocks: " << blocks << ", " 
              << "threads: (" << threads.x << ", " 
              << threads.y << ", " << threads.z << ")" << std::endl;


    cuda_naive_sbvr_mm_T<LIndexT, RIndexT><<<blocks, threads>>>(
        l_bvr, (LIndexT*)l_coeff_idx, l_coeff_cache,
        r_bvr, (RIndexT*)r_coeff_idx, r_coeff_cache,
        bias, out,
        M, N, K,
        l_num_sums, r_num_sums);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

typedef void (*KernelLaunchFn)(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K);

template <typename LIndexT, typename RIndexT, int L_NUM_SUMS, int R_NUM_SUMS>
void launch_8x4_sbvr_kernel(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount * 8;
    dim3 threads = 32;

    std::cout << "Launching 8x4 SBVR kernel <" 
              << typeid(LIndexT).name() << ", " 
              << typeid(RIndexT).name() << ", "
              << "l_num_sums: " << L_NUM_SUMS << ", "
              << "r_num_sums: " << R_NUM_SUMS << ", "
              << "M: " << M << ", "
              << "N: " << N << ", "
              << "K: " << K << ", "
              << "blocks: " << blocks << ", " 
              << "threads: (" << threads.x << ", " 
              << threads.y << ", " << threads.z << ")" << std::endl;

    cuda_8x4_sbvr_mm_T<LIndexT, RIndexT, 
                        L_NUM_SUMS, R_NUM_SUMS><<<blocks, threads>>>(
        l_bvr, (LIndexT*)l_coeff_idx, l_coeff_cache,
        r_bvr, (RIndexT*)r_coeff_idx, r_coeff_cache,
        bias, out,
        M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

template <typename LIndexT, typename RIndexT>
void launch_8x4_sbvr_kernel_wrapper(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums)
{
    KernelLaunchFn kernel_list[] = {
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 4, 4>,
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 4, 6>,
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 4, 8>,
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 6, 4>,
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 6, 6>,
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 6, 8>,
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 8, 4>,
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 8, 6>,
        launch_8x4_sbvr_kernel<LIndexT, RIndexT, 8, 8>
    };
    int kernel_idx = (l_num_sums - 4)/2 * 3 + (r_num_sums - 4)/2;
    if (kernel_idx < 0 || kernel_idx >= 9)
    {
        std::cerr << "Invalid kernel index: " << kernel_idx << std::endl;
        throw std::runtime_error("Invalid kernel index");
    }
    kernel_list[kernel_idx](
           l_bvr, l_coeff_idx, l_coeff_cache,
           r_bvr, r_coeff_idx, r_coeff_cache,
           bias, out,
           M, N, K);
}

template <typename LIndexT, typename RIndexT>
void launch_coeff_idx_typed_sbvr_kernel(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums)
{
    bool supported_num_sums = 
        (l_num_sums == 4 || l_num_sums == 6 || l_num_sums == 8) &&
        (r_num_sums == 4 || r_num_sums == 6 || r_num_sums == 8);

    if (supported_num_sums && M % 8 == 0 && N % 4 == 0)
    {
        launch_8x4_sbvr_kernel_wrapper<LIndexT, RIndexT>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out,
            M, N, K,
            l_num_sums, r_num_sums);
    }
    else
    {
        launch_naive_sbvr_kernel<LIndexT, RIndexT>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out,
            M, N, K,
            l_num_sums, r_num_sums);
    }
}

void launch_cuda_sbvr_mm_T(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx,__half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size)
{

    const bool use_l_uint8 = (l_cache_size <= 256);
    const bool use_r_uint8 = (r_cache_size <= 256);

    if (use_l_uint8 && use_r_uint8)
    {
        launch_coeff_idx_typed_sbvr_kernel<uint8_t, uint8_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out,
            M, N, K,
            l_num_sums, r_num_sums);
    }
    else if (use_l_uint8 && !use_r_uint8)
    {
        launch_coeff_idx_typed_sbvr_kernel<uint8_t, uint16_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out,
            M, N, K,
            l_num_sums, r_num_sums);
    }
    else if (!use_l_uint8 && use_r_uint8)
    {
        launch_coeff_idx_typed_sbvr_kernel<uint16_t, uint8_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out,
            M, N, K,
            l_num_sums, r_num_sums);
    }
    else
    {
        launch_coeff_idx_typed_sbvr_kernel<uint16_t, uint16_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out,
            M, N, K,
            l_num_sums, r_num_sums);
    }
}