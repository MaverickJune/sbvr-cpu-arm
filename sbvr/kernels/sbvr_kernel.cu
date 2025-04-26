
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>

#define T_BLOCK_PER_SM 24
#define K_PER_BVR 8 // BVR size 256
#define _1xtN_TILE_N_SIZE 4
#define THREAD_PER_WARP 32
#define T_BLOCK_TILE_SIZE 32

typedef void (*KernelLaunchFn)(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int device_id);

extern int device_count;
extern cudaDeviceProp cuda_prop_list[16];

template <typename LIndexT, typename RIndexT>
__global__ void cuda_naive_sbvr_mm_T(
    uint32_t* l_bvr, LIndexT* l_coeff_idx, __half* __restrict__ l_coeff_cache,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums)
{
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
            int l_coeff_cache_idx = l_coeff_idx[bvr_idx * M + m];
            int r_coeff_cache_idx = r_coeff_idx[bvr_idx * N + n];
            __half* l_coeff_ptr = 
                            &l_coeff_cache[l_coeff_cache_idx * l_num_sums];
            __half* r_coeff_ptr = 
                            &r_coeff_cache[r_coeff_cache_idx * r_num_sums];
            #pragma unroll
            for (int k = 0; k < K_PER_BVR; k++)
            {
                const int k_idx = bvr_idx * K_PER_BVR + k;   
                for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
                {
                    uint32_t l;
                    if ((l_num_sums & 1) == 0)
                        l = l_bvr[(l_idx / 2) * K * M * 2 + 
                                    (k_idx * M + m) * 2 + (l_idx % 2)];
                    else
                        l = l_bvr[l_idx * K * M + k_idx * M + m];
                    const float l_coeff = __half2float(l_coeff_ptr[l_idx]);

                    for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                    {
                        uint32_t r;
                        if ((r_num_sums & 1) == 0)
                            r = r_bvr[(r_idx / 2) * K * N * 2 + 
                                        (k_idx * N + n) * 2 + (r_idx % 2)];
                        else
                            r = r_bvr[r_idx * K * N + k_idx * N + n];
                        const float r_coeff = __half2float(r_coeff_ptr[r_idx]);
                        const uint32_t lr = l & r;
                        const float lr_popc = (float)__popc(lr);
                        sum += lr_popc * l_coeff * r_coeff;
                        // if (tid == 0)
                        //     printf("bvr_idx: %d, l: %u, r: %u, lr: %u, "
                        //         "lr_popc: %f, coeff_mult: %f, sum: %f\n", 
                        //         bvr_idx, l, r, lr, lr_popc, 
                        //         coeff_mult[l_idx][r_idx], sum);
                    }
                }
            }
        }
        out[i] = __float2half(sum);
    }
    
}

template <typename LIndexT, typename RIndexT, int L_NUM_SUMS, int R_NUM_SUMS,
          int TILE_M, int TILE_N>
__global__ void cuda_tMxtN_sbvr_mm_T(
    uint32_t* l_bvr, LIndexT* l_coeff_idx, __half* __restrict__ l_coeff_cache,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int M, int N, int K)
{
    // Tensor shapes:
    // l_bvr: [self.num_sums / 2, K, M, 2]
    // l_coeff_idx: [num_bvr_per_K, M]
    // l_coeff_cache: [cache_size, L_NUM_SUMS]
    // r_bvr: [self.num_sums / 2, K, N, 2]
    // r_coeff_idx: [num_bvr_per_K, N]
    // r_coeff_cache: [cache_size, R_NUM_SUMS]

    __half coeff_mult[L_NUM_SUMS][R_NUM_SUMS];

    const int num_tblock_m = M / T_BLOCK_TILE_SIZE;
    const int num_tblock_n = N / T_BLOCK_TILE_SIZE;
    const int num_tblock_tiles = num_tblock_m * num_tblock_n;
    const int bvr_per_K = K / K_PER_BVR;
    const int tid = threadIdx.x * blockDim.y + threadIdx.y;
    const int g_tid = blockIdx.x * (blockDim.x * blockDim.y) + tid;

    for (int tblock_id = blockIdx.x; tblock_id < num_tblock_tiles;
         tblock_id += gridDim.x)
    {
        const int tblock_m = (tblock_id / num_tblock_n) * T_BLOCK_TILE_SIZE;
        const int tblock_n = (tblock_id % num_tblock_n) * T_BLOCK_TILE_SIZE;
        const int m = tblock_m + threadIdx.x * TILE_M;
        const int n = tblock_n + threadIdx.y * TILE_N;
        
        if (tid == 0)
            printf("gTid %d Tid %d) Tblock_id %d, M: %d, N: %d, m: %d, n: %d, "
                   "l_num_sums: %d, r_num_sums: %d, bvr_per_K: %d\n",
                   g_tid, tid, tblock_id, M, N, m, n,
                   L_NUM_SUMS, R_NUM_SUMS, bvr_per_K);
    }
    
    
}

template <typename LIndexT, typename RIndexT, int L_NUM_SUMS, int R_NUM_SUMS,
          int TILE_N>
__global__ void cuda_1xtN_sbvr_mm_T(
    uint32_t* l_bvr, LIndexT* l_coeff_idx, __half* __restrict__ l_coeff_cache,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int M, int N, int K)
{
    // Tensor shapes:
    // l_bvr: [self.num_sums / 2, K, M, 2]
    // l_coeff_idx: [num_bvr_per_K, M]
    // l_coeff_cache: [cache_size, L_NUM_SUMS]
    // r_bvr: [self.num_sums / 2, K, N, 2]
    // r_coeff_idx: [num_bvr_per_K, N]
    // r_coeff_cache: [cache_size, R_NUM_SUMS]

    const int tblock_per_N = (N + TILE_N - 1) / TILE_N;
    const int bvr_per_K = K / K_PER_BVR;
    // const int g_tid = blockIdx.x * THREAD_PER_WARP + 
    //                     threadIdx.y * TILE_N + threadIdx.x;
    
    for (int tblock_id = blockIdx.x; tblock_id < tblock_per_N * M;
         tblock_id += gridDim.x)
    {
        const int n = (tblock_id / M) * TILE_N + threadIdx.x;
        const int m = (tblock_id % M);
        __half2 sum = __float2half2_rn(0.0f);
        if (n < N)
        {
            
            // if (g_tid == 0)
            //     printf("gTid %d) bi.x:%d, ti.x: %d, ti.y: %d, "
            //            "Tblock_id %d, M: %d, N: %d, m: %d, n: %d, "
            //            "l_num_sums: %d, r_num_sums: %d, bvr_per_K: %d\n",
            //            g_tid, blockIdx.x, threadIdx.x, threadIdx.y,
            //            tblock_id, M, N, m, n,
            //            L_NUM_SUMS, R_NUM_SUMS, bvr_per_K);

            for (int bvr_idx = threadIdx.y; bvr_idx < bvr_per_K; 
                    bvr_idx += blockDim.y)
            {
                int2 popc_cache [L_NUM_SUMS / 2][R_NUM_SUMS] = {};
                #pragma unroll
                for (int k = 0; k < K_PER_BVR; k++)
                {
                    const int k_idx = bvr_idx * K_PER_BVR + k;   
                    const int2* l_bvr_ptr = (int2*)&l_bvr[(k_idx * M + m) * 2];
                    const int2* r_bvr_ptr = (int2*)&r_bvr[(k_idx * N + n) * 2];
                    #pragma unroll
                    for (int l_idx = 0; l_idx < L_NUM_SUMS / 2; l_idx++)
                    {
                        const int2 l_0_1 =l_bvr_ptr[l_idx * K * M];
                        #pragma unroll
                        for (int r_idx = 0; r_idx < R_NUM_SUMS / 2; r_idx++)
                        {
                            const int2 r_0_1 =r_bvr_ptr[r_idx * K * N];
                            popc_cache[l_idx][r_idx * 2].x += 
                                                __popc(((uint32_t)(l_0_1.x)) & 
                                                    ((uint32_t)(r_0_1.x)));
                            popc_cache[l_idx][r_idx * 2].y += 
                                                __popc(((uint32_t)(l_0_1.y)) & 
                                                    ((uint32_t)(r_0_1.y)));
                            popc_cache[l_idx][r_idx * 2 + 1].x += 
                                                __popc(((uint32_t)(l_0_1.x)) & 
                                                    ((uint32_t)(r_0_1.y)));
                            popc_cache[l_idx][r_idx * 2 + 1].y += 
                                                __popc(((uint32_t)(l_0_1.y)) & 
                                                    ((uint32_t)(r_0_1.x)));
                            // if (g_tid == 0)
                            //     printf("bvr_idx: %d, kidx: %d, l_idx: %d, "
                            //            "r_idx: %d, l_0_1: (%d, %d), "
                            //            "r_0_1: (%d, %d), lr_00_11: (%f, %f), "
                            //            "lr_01_10: (%f, %f), co_mul_00_11: (%f, %f), "
                            //            "co_mul_01_10: (%f, %f), sum: (%f, %f)\n",
                            //            bvr_idx, k_idx, l_idx, r_idx,
                            //            l_0_1.x, l_0_1.y,
                            //            r_0_1.x, r_0_1.y,
                            //            __half2float(lr_00_11.x), 
                            //            __half2float(lr_00_11.y),
                            //            __half2float(lr_01_10.x), 
                            //            __half2float(lr_01_10.y),
                            //            __half2float(
                            //                     coeff_mult[l_idx][r_idx * 2].x),
                            //            __half2float(
                            //                     coeff_mult[l_idx][r_idx * 2].y),
                            //            __half2float(
                            //                     coeff_mult[l_idx][r_idx * 2 + 1].x),
                            //            __half2float(
                            //                     coeff_mult[l_idx][r_idx * 2 + 1].y),
                            //            __half2float(sum.x), 
                            //            __half2float(sum.y));
                        }
                    }
                }
                const __half2* l_coeff_ptr = 
                        (__half2*)&l_coeff_cache[l_coeff_idx[bvr_idx * M + m] 
                                                    * L_NUM_SUMS];
                const __half2* r_coeff_ptr = 
                        (__half2*)&r_coeff_cache[r_coeff_idx[bvr_idx * N + n] 
                                                    * R_NUM_SUMS];
                #pragma unroll
                for (int l_idx = 0; l_idx < L_NUM_SUMS / 2; l_idx++)
                {
                    #pragma unroll
                    for (int r_idx = 0; r_idx < R_NUM_SUMS / 2; r_idx++)
                    {
                        const __half2 popc_h_0 = 
                            __halves2half2(
                                __float2half(
                                    (float)popc_cache[l_idx][r_idx * 2].x),
                                __float2half(
                                    (float)popc_cache[l_idx][r_idx * 2].y));
                        const __half2 popc_h_1 = 
                            __halves2half2(
                                __float2half(
                                    (float)popc_cache[l_idx][r_idx * 2 + 1].x),
                                __float2half(
                                    (float)popc_cache[l_idx][r_idx * 2 + 1].y));
                        const __half2 coeff_0 = 
                            __hmul2(l_coeff_ptr[l_idx], r_coeff_ptr[r_idx]);
                        const __half2 coeff_1 =
                            __hmul2(l_coeff_ptr[l_idx], 
                                __halves2half2(__high2half(r_coeff_ptr[r_idx]), 
                                            __low2half(r_coeff_ptr[r_idx])));        
                        sum = __hfma2(coeff_0, popc_h_0, sum);
                        sum = __hfma2(coeff_1, popc_h_1, sum);

                    }
                }
            }
            
            // printf("\tgTid %d) tblock_id: %d, m: %d, n: %d, sum: (%f, %f)\n", 
            //             g_tid, tblock_id, m, n, __half2float(sum.x), 
            //             __half2float(sum.y));
            
        }
        // Reduce the sum across blockDim.y
        #pragma unroll
        for (int i = (THREAD_PER_WARP / TILE_N) / 2; i > 0; i /= 2)
        {
            __half2 rec = __shfl_down_sync(0xFFFFFFFF, 
                                         sum, i * TILE_N);
            sum = __hadd2(sum, rec);
            // if (threadIdx.y == 0)
            //     printf("\tgTid %d) tblock_id: %d, m: %d, n: %d, stride %d, " 
            //             "rec (%f, %f), sum: (%f, %f)\n", 
            //                 g_tid, tblock_id, m, n, i * TILE_N,
            //                 __half2float(rec.x), __half2float(rec.y), 
            //                 __half2float(sum.x), __half2float(sum.y));
        }
        // Store the result in the output matrix
        if (threadIdx.y == 0 && n < N)
        {
            __half bias_val = (bias != nullptr ? bias[n] : __float2half(0.0f));
            sum.x = __hadd(sum.x, sum.y);  
            sum.x = __hadd(sum.x, bias_val); 
            out[m * N + n] = sum.x; 
            // if (g_tid == 0)
            //     printf("\tgTid %d) tblock_id: %d, m: %d, n: %d, sum: %f\n", 
            //         g_tid, tblock_id, m, n, __half2float(sum.x));
        }
    }
}

template <typename LIndexT, typename RIndexT>
void launch_naive_sbvr_kernel(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int device_id = 0)
{
    int blocks = cuda_prop_list[device_id].multiProcessorCount * T_BLOCK_PER_SM;
    dim3 threads = 32;

    // std::cout << "Launching naive SBVR kernel <" 
    //           << typeid(LIndexT).name() << ", " 
    //           << typeid(RIndexT).name() << ", "
    //           << "l_num_sums: " << l_num_sums << ", "
    //           << "r_num_sums: " << r_num_sums << ", "
    //           << "M: " << M << ", "
    //           << "N: " << N << ", "
    //           << "K: " << K << ", "
    //           << "blocks: " << blocks << ", " 
    //           << "threads: (" << threads.x << ", " 
    //           << threads.y << ", " << threads.z << ")" << std::endl;


    cuda_naive_sbvr_mm_T<LIndexT, RIndexT><<<blocks, threads>>>(
        l_bvr, (LIndexT*)l_coeff_idx, l_coeff_cache,
        r_bvr, (RIndexT*)r_coeff_idx, r_coeff_cache,
        bias, out,
        M, N, K,
        l_num_sums, r_num_sums
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

template <typename LIndexT, typename RIndexT, int L_NUM_SUMS, int R_NUM_SUMS,
            int TILE_M, int TILE_N>
void launch_tMxtN_sbvr_kernel(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int device_id = 0)
{
    int blocks = cuda_prop_list[device_id].multiProcessorCount * T_BLOCK_PER_SM;
    dim3 threads = {T_BLOCK_TILE_SIZE / TILE_M, 
                    T_BLOCK_TILE_SIZE / TILE_N, 1};

    // std::cout << "Launching " << TILE_M << "x" << TILE_N << " SBVR kernel <" 
    //           << typeid(LIndexT).name() << ", " 
    //           << typeid(RIndexT).name() << ", "
    //           << "l_num_sums: " << L_NUM_SUMS << ", "
    //           << "r_num_sums: " << R_NUM_SUMS << ", "
    //           << "M: " << M << ", "
    //           << "N: " << N << ", "
    //           << "K: " << K << ", "
    //           << "blocks: " << blocks << ", " 
    //           << "threads: (" << threads.x << ", " 
    //           << threads.y << ", " << threads.z << ")" << std::endl;

    cuda_tMxtN_sbvr_mm_T<LIndexT, RIndexT, L_NUM_SUMS, R_NUM_SUMS, 
        TILE_M, TILE_N> <<<blocks, threads>>>(
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
void launch_tMxtN_sbvr_kernel_wrapper(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int device_id = 0)
{
    KernelLaunchFn kernel_list[] = {
        // <LIndexT, RIndexT, L_NUM_SUMS, R_NUM_SUMS, TILE_M, TILE_N>
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 2, 2, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 2, 4, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 2, 6, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 2, 8, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 2, 10, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 4, 2, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 4, 4, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 4, 6, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 4, 8, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 4, 10, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 6, 2, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 6, 4, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 6, 6, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 6, 8, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 6, 10, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 8, 2, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 8, 4, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 8, 6, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 8, 8, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 8, 10, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 10, 2, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 10, 4, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 10, 6, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 10, 8, 8, 4>,
        launch_tMxtN_sbvr_kernel<LIndexT, RIndexT, 10, 10, 8, 4>,
    };
    int kernel_idx = (l_num_sums - 2)/2 * 5 + (r_num_sums - 2)/2;
    if (kernel_idx < 0 || kernel_idx > 25)
    {
        std::cerr << "Invalid kernel index: " << kernel_idx << std::endl;
        throw std::runtime_error("Invalid kernel index");
    }
    kernel_list[kernel_idx](
           l_bvr, l_coeff_idx, l_coeff_cache,
           r_bvr, r_coeff_idx, r_coeff_cache,
           bias, out,
           M, N, K,
           device_id);
}

template <typename LIndexT, typename RIndexT, int L_NUM_SUMS, int R_NUM_SUMS,
            int TILE_N>
void launch_1xtN_sbvr_kernel(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int device_id = 0)
{
    // int blocks = 1;
    int blocks = cuda_prop_list[device_id].multiProcessorCount * T_BLOCK_PER_SM;
    dim3 threads = {TILE_N, THREAD_PER_WARP / TILE_N};

    // std::cout << "Launching " << 1 << "x" << TILE_N << " SBVR kernel <" 
    //           << typeid(LIndexT).name() << ", " 
    //           << typeid(RIndexT).name() << ", "
    //           << "l_num_sums: " << L_NUM_SUMS << ", "
    //           << "r_num_sums: " << R_NUM_SUMS << ", "
    //           << "M: " << M << ", "
    //           << "N: " << N << ", "
    //           << "K: " << K << ", "
    //           << "blocks: " << blocks << ", " 
    //           << "threads: (" << threads.x << ", " 
    //           << threads.y << ", " << threads.z << ")" << std::endl;

    cuda_1xtN_sbvr_mm_T<LIndexT, RIndexT, L_NUM_SUMS, R_NUM_SUMS, 
        TILE_N> <<<blocks, threads>>>(
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
void launch_1xtN_sbvr_kernel_wrapper(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int device_id = 0)
{
    KernelLaunchFn kernel_list[] = {
        // <LIndexT, RIndexT, L_NUM_SUMS, R_NUM_SUMS, TILE_N>
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 2, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 4, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 6, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 8, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 10, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 2, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 4, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 6, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 8, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 10, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 2, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 4, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 6, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 8, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 10, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 2, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 4, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 6, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 8, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 10, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 2, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 4, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 6, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 8, _1xtN_TILE_N_SIZE>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 10, _1xtN_TILE_N_SIZE>,
    };
    int kernel_idx = (l_num_sums - 2)/2 * 5 + (r_num_sums - 2)/2;
    kernel_list[kernel_idx](
           l_bvr, l_coeff_idx, l_coeff_cache,
           r_bvr, r_coeff_idx, r_coeff_cache,
           bias, out,
           M, N, K,
           device_id);
}

void launch_cuda_sbvr_mm_T(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx,__half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size,
    int device_id = 0)
{
    // printf("Shared memory per block: %zu bytes\n", cuda_prop.sharedMemPerBlock);
    // printf("Shared memory per SM: %zu bytes\n", cuda_prop.sharedMemPerMultiprocessor);

    // std::cout << "sbvr_mm_T: M=" << M << ", N=" << N << ", K=" << K
    // << ", l_num_sums=" << l_num_sums << ", r_num_sums=" << r_num_sums
    // << ", l_cache_size=" << l_cache_size << ", r_cache_size=" << r_cache_size
    // << std::endl;

    const bool use_l_uint8 = (l_cache_size <= 256);
    const bool use_r_uint8 = (r_cache_size <= 256);
    const bool supported_num_sums = (l_num_sums & 1) == 0 && 
                                    (l_num_sums <= 10) &&
                                    (r_num_sums & 1) == 0 && 
                                    (r_num_sums <= 10);
    if (supported_num_sums)
    {
        if (use_l_uint8 && use_r_uint8)
        {
            launch_1xtN_sbvr_kernel_wrapper<uint8_t, uint8_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                device_id);
        }
        else if (use_l_uint8 && !use_r_uint8)
        {
            launch_1xtN_sbvr_kernel_wrapper<uint8_t, uint16_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                device_id);
        }
        else if (!use_l_uint8 && use_r_uint8)
        {
            launch_1xtN_sbvr_kernel_wrapper<uint16_t, uint8_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                device_id);
        }
        else
        {
            launch_1xtN_sbvr_kernel_wrapper<uint16_t, uint16_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                device_id);
        }
    }
    else
    {
        if (use_l_uint8 && use_r_uint8)
        {
            launch_naive_sbvr_kernel<uint8_t, uint8_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                device_id);
        }
        else if (use_l_uint8 && !use_r_uint8)
        {
            launch_naive_sbvr_kernel<uint8_t, uint16_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                device_id);
        }
        else if (!use_l_uint8 && use_r_uint8)
        {
            launch_naive_sbvr_kernel<uint16_t, uint8_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                device_id);
        }
        else
        {
            launch_naive_sbvr_kernel<uint16_t, uint16_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                device_id);
        }
    }
}