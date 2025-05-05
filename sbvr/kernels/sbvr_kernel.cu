#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>

#define BLOCK_PER_SM 16
#define K_PER_BVR 8 // BVR size 256
#define _1xtN_tN 4
#define THREAD_PER_WARP 32
#define WARP_PER_BLOCK 4
#define BLOCK_TILE_SIZE 32

// For row_deq_mm_T
#define N_PER_BVR 8

template <int NUM_SUMS>
struct bvrs;

template <>
struct bvrs<2> {
    int2 data;
    __device__ __forceinline__ int get(int idx) const 
    {
        return idx == 0 ? data.x : data.y;
    }
};
template <>
struct bvrs<4> {
    int4 data;
    __device__ __forceinline__ int get(int idx) const 
    {
        if (idx == 0) return data.x;
        else if (idx == 1) return data.y;
        else if (idx == 2) return data.z;
        else return data.w;
    }
};
template <>
struct bvrs<6> {
    int2 data0;
    int2 data1;
    int2 data2;
    __device__ __forceinline__ int get(int idx) const 
    {
        if (idx == 0) return data0.x;
        else if (idx == 1) return data0.y;
        else if (idx == 2) return data1.x;
        else if (idx == 3) return data1.y;
        else if (idx == 4) return data2.x;
        else return data2.y;
    }
};
template <>
struct bvrs<8> {
    int4 data0;
    int4 data1;
    __device__ __forceinline__ int get(int idx) const 
    {
        if (idx == 0) return data0.x;
        else if (idx == 1) return data0.y;
        else if (idx == 2) return data0.z;
        else if (idx == 3) return data0.w;
        else if (idx == 4) return data1.x;
        else if (idx == 5) return data1.y;
        else if (idx == 6) return data1.z;
        else return data1.w;
    }
};
template <>
struct bvrs<10> {
    int2 data0;
    int2 data1;
    int2 data2;
    int2 data3;
    int2 data4;
    __device__ __forceinline__ int get(int idx) const 
    {
        if (idx == 0) return data0.x;
        else if (idx == 1) return data0.y;
        else if (idx == 2) return data1.x;
        else if (idx == 3) return data1.y;
        else if (idx == 4) return data2.x;
        else if (idx == 5) return data2.y;
        else if (idx == 6) return data3.x;
        else if (idx == 7) return data3.y;
        else if (idx == 8) return data4.x;
        else return data4.y;
    }
};

template <int NUM_SUMS>
struct coeffs {
    __half2 coeff[NUM_SUMS / 2];
};

typedef void (*KernelLaunchFn)(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int device_id);

typedef void (*RDKernelLaunchFN)(
    __half* l_w,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int M, int N, int K,
    int use_shfl,
    int device_id);

extern int device_count;
extern cudaDeviceProp cuda_prop_list[16];

template <typename RIndexT, int RNumSums>
__global__ void cuda_row_deq_mm_T(
    __half* __restrict__ l_w,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int M, int N, int K)
{
    /*
    C = A @ B^T
    r_bvr is grouped in row-direction: (N/32(num_bits in bvr dtype, uint32), K, num_sums)
    r_coeff_idx: (N/group_size, K)
    */
   
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = tid >> 5;
    const int N_EFF = N / 32; 
    const int totalWarps = M * N_EFF;

    for (int wid = warp_id; wid < totalWarps; wid += (blockDim.x * gridDim.x)>>5)
    {
        int m = wid / N_EFF;
        int n = wid % N_EFF; // which group of 32 in N

        float sum = (bias != nullptr) ? __half2float(bias[n * 32 + lane]) : 0.0f;

        __half l_val;
        uint32_t bv[RNumSums];
        uint32_t curr_r_coeff_idx;

        #pragma unroll
        for (int k = 0; k < K; k++)
        {
            if (lane == 0)
            {
                l_val = l_w[m * K + k];
                #pragma unroll
                for (int s = 0; s < RNumSums; s++)
                    bv[s] = r_bvr[(n * K + k) * RNumSums + s];
                curr_r_coeff_idx = r_coeff_idx[(n / N_PER_BVR) * K + k];
            }

            float c_board[RNumSums];
            uint32_t b_board[RNumSums];

            constexpr unsigned FULL = 0xFFFFFFFFu;
            l_val = __shfl_sync(FULL, l_val, 0);
            curr_r_coeff_idx = __shfl_sync(FULL, curr_r_coeff_idx, 0);

            #pragma unroll
            for (int s = 0; s < RNumSums; s++)
                bv[s] = __shfl_sync(FULL, bv[s], 0);
            
            float l_val_f = __half2float(l_val);
            #pragma unroll
            for(int s = 0; s < RNumSums; s++)
            {
                c_board[s] = __half2float(r_coeff_cache[curr_r_coeff_idx * RNumSums + s]);
                b_board[s] = (bv[s] >> lane) & 1;
            }

            float dot = 0.0f;
            #pragma unroll
            for (int s = 0; s < RNumSums; s++) {
                dot = __fmaf_rn(c_board[s], (float)b_board[s], dot);
            }
            sum = __fmaf_rn(l_val_f, dot, sum);
        }
        out[m * N + (n * 32 + lane)] = __float2half(sum);
    }
}

template <typename RIndexT, int RNumSums>
__global__ void cuda_row_deq_wo_shfl_mm_T(
    __half* __restrict__ l_w,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int M, int N, int K)
{
    /*
    C = A @ B^T
    r_bvr is grouped in row-direction: (N/32(num_bits in bvr dtype, uint32), K, num_sums)
    r_coeff_idx: (N/group_size, K)
    In this kernel, expect that same memory addr access within a warp will fall back into a single L1 fetch
    and broadcast
    */
   
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = tid >> 5;
    const int N_EFF = N / 32; 
    const int totalWarps = M * N_EFF;

    for (int wid = warp_id; wid < totalWarps; wid += (blockDim.x * gridDim.x)>>5)
    {
        int m = wid / N_EFF;
        int n = wid % N_EFF; // which group of 32 in N

        float sum = (bias != nullptr) ? __half2float(bias[n * 32 + lane]) : 0.0f;

        __half l_val;
        uint32_t bv[RNumSums];
        uint32_t curr_r_coeff_idx;

        #pragma unroll
        for (int k = 0; k < K; k++)
        {
            l_val = l_w[m * K + k];
            #pragma unroll
            for (int s = 0; s < RNumSums; s++)
                bv[s] = r_bvr[(n * K + k) * RNumSums + s];
            curr_r_coeff_idx = r_coeff_idx[(n / N_PER_BVR) * K + k];

            float c_board[RNumSums];
            uint32_t b_board[RNumSums];
            
            float l_val_f = __half2float(l_val);
            #pragma unroll
            for(int s = 0; s < RNumSums; s++)
            {
                c_board[s] = __half2float(r_coeff_cache[curr_r_coeff_idx * RNumSums + s]);
                b_board[s] = (bv[s] >> lane) & 1;
            }

            float dot = 0.0f;
            #pragma unroll
            for (int s = 0; s < RNumSums; s++) {
                dot = __fmaf_rn(c_board[s], (float)b_board[s], dot);
            }
            sum = __fmaf_rn(l_val_f, dot, sum);
        }
        out[m * N + (n * 32 + lane)] = __float2half(sum);
    }
}

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
                    const uint32_t l = 
                            l_bvr[(k_idx * M + m) * l_num_sums + l_idx];
                    const float l_coeff = __half2float(l_coeff_ptr[l_idx]);
                    for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                    {
                        const uint32_t r = 
                                r_bvr[(k_idx * N + n) * r_num_sums + r_idx];
                        const float r_coeff = __half2float(r_coeff_ptr[r_idx]);
                        const uint32_t lr = l & r;
                        const float lr_popc = (float)__popc(lr);
                        sum += lr_popc * l_coeff * r_coeff;
                    }
                }
            }
        }
        out[i] = __float2half(sum);
    }
}

template <typename LIndexT, typename RIndexT, int LNumSums, int RNumSums,
          int TileM, int TileN>
__global__ void cuda_tMxtN_sbvr_mm_T(
    uint32_t* l_bvr, LIndexT* l_coeff_idx, __half* __restrict__ l_coeff_cache,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int M, int N, int K)
{
    // Tensor shapes:
    // l_bvr: [self.num_sums / 2, K, M, 2]
    // l_coeff_idx: [num_bvr_per_K, M]
    // l_coeff_cache: [cache_size, LNumSums]
    // r_bvr: [self.num_sums / 2, K, N, 2]
    // r_coeff_idx: [num_bvr_per_K, N]
    // r_coeff_cache: [cache_size, RNumSums]

    __half coeff_mult[LNumSums][RNumSums];

    const int num_tblock_m = M / BLOCK_TILE_SIZE;
    const int num_tblock_n = N / BLOCK_TILE_SIZE;
    const int num_tblock_tiles = num_tblock_m * num_tblock_n;
    const int bvr_per_K = K / K_PER_BVR;
    const int tid = threadIdx.x * blockDim.y + threadIdx.y;
    const int g_tid = blockIdx.x * (blockDim.x * blockDim.y) + tid;

    for (int tblock_id = blockIdx.x; tblock_id < num_tblock_tiles;
         tblock_id += gridDim.x)
    {
        const int tblock_m = (tblock_id / num_tblock_n) * BLOCK_TILE_SIZE;
        const int tblock_n = (tblock_id % num_tblock_n) * BLOCK_TILE_SIZE;
        const int m = tblock_m + threadIdx.x * TileM;
        const int n = tblock_n + threadIdx.y * TileN;
        
        if (tid == 0)
            printf("gTid %d Tid %d) Tblock_id %d, M: %d, N: %d, m: %d, n: %d, "
                   "l_num_sums: %d, r_num_sums: %d, bvr_per_K: %d\n",
                   g_tid, tid, tblock_id, M, N, m, n,
                   LNumSums, RNumSums, bvr_per_K);
    }
    
    
}

template <typename LIndexT, typename RIndexT, int LNumSums, int RNumSums,
          int TileN>
__global__ void cuda_1xtN_sbvr_mm_T(
    uint32_t* l_bvr, LIndexT* l_coeff_idx, __half* __restrict__ l_coeff_cache,
    uint32_t* r_bvr, RIndexT* r_coeff_idx, __half* __restrict__ r_coeff_cache,
    __half* __restrict__ bias, __half* __restrict__ out,
    int M, int N, int K)
{
    const int tblock_per_N = (N + TileN - 1) / TileN;
    const int bvr_per_K = K / K_PER_BVR;

    for (int tblock_id = blockIdx.x * blockDim.z + threadIdx.z; 
        tblock_id < tblock_per_N * M;
         tblock_id += gridDim.x * blockDim.z)
    {
        const int n = (tblock_id / M) * TileN + threadIdx.x;
        const int m = (tblock_id % M);
        float sum = 0.0f;
        if (n < N)
        {
            for (int bvr_idx = threadIdx.y; bvr_idx < bvr_per_K; 
                    bvr_idx += blockDim.y)
            {
                // If all bits of l_bvr and r_bvr are set, this may overflow.
                uchar4 popc_cache [LNumSums / 2][RNumSums / 2] = {};
                #pragma unroll
                for (int k = 0; k < K_PER_BVR; k++)
                {
                    const int k_idx = bvr_idx * K_PER_BVR + k;   
                    const bvrs<LNumSums> l_bvrs = 
                        *(bvrs<LNumSums>*)(&l_bvr[(k_idx * M + m) * LNumSums]);
                    const bvrs<RNumSums> r_bvrs = 
                        *(bvrs<RNumSums>*)(&r_bvr[(k_idx * N + n) * RNumSums]);
                    #pragma unroll
                    for (int l_idx = 0; l_idx < LNumSums / 2; l_idx++)
                    {
                        #pragma unroll
                        for (int r_idx = 0; r_idx < RNumSums / 2; r_idx++)
                        {
                            const uint32_t l_0 = l_bvrs.get(l_idx * 2);
                            const uint32_t l_1 = l_bvrs.get(l_idx * 2 + 1);
                            const uint32_t r_0 = r_bvrs.get(r_idx * 2);
                            const uint32_t r_1 = r_bvrs.get(r_idx * 2 + 1);
                            popc_cache[l_idx][r_idx].x += __popc(l_0 & r_0);
                            popc_cache[l_idx][r_idx].y += __popc(l_1 & r_1);
                            popc_cache[l_idx][r_idx].z += __popc(l_0 & r_1);
                            popc_cache[l_idx][r_idx].w += __popc(l_1 & r_0);
                        }
                    }
                }
                const int l_coeff_i = __ldg(&l_coeff_idx[bvr_idx * M + m]);
                const int r_coeff_i = __ldg(&r_coeff_idx[bvr_idx * N + n]);
                const coeffs<LNumSums> l_coeffs = 
                    *(coeffs<LNumSums>*)(&l_coeff_cache[l_coeff_i * LNumSums]);
                const coeffs<RNumSums> r_coeffs = 
                    *(coeffs<RNumSums>*)(&r_coeff_cache[r_coeff_i * RNumSums]);
                #pragma unroll
                for (int l_idx = 0; l_idx < LNumSums / 2; l_idx++)
                {
                    #pragma unroll
                    for (int r_idx = 0; r_idx < RNumSums / 2; r_idx++)
                    {
                        const __half2 popc_h_0 = 
                            __halves2half2(
                                __ushort2half_rd(
                                    (ushort)popc_cache[l_idx][r_idx].x),
                                __ushort2half_rd(
                                    (ushort)popc_cache[l_idx][r_idx].y));
                        const __half2 popc_h_1 = 
                            __halves2half2(
                                __ushort2half_rd(
                                    (ushort)popc_cache[l_idx][r_idx].z),
                                __ushort2half_rd(
                                    (ushort)popc_cache[l_idx][r_idx].w));
                        const __half2 l_coeff = l_coeffs.coeff[l_idx];
                        const __half2 r_coeff = r_coeffs.coeff[r_idx];
                        const __half2 coeff_0 = __hmul2(l_coeff, r_coeff);
                        const __half2 coeff_1 =
                            __hmul2(l_coeff, 
                                        __halves2half2(__high2half(r_coeff), 
                                                       __low2half(r_coeff)));
                        const __half2 mult_sum = __hfma2(coeff_0, popc_h_0, 
                                                    __hmul2(coeff_1, popc_h_1));                               
                        sum += __half2float(mult_sum.x) + 
                               __half2float(mult_sum.y);
                    }
                }
            }
        }
        #pragma unroll
        for (int i = (THREAD_PER_WARP / TileN) / 2; i > 0; i /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, i * TileN);

        if (threadIdx.y == 0 && n < N)
        {
            __half bias_val = (bias != nullptr ? bias[n] : __float2half(0.0f));
            bias_val = __hadd(__float2half(sum), bias_val);  
            out[m * N + n] = bias_val; 
        }
    }
}

template <typename RIndexT, int RNumSums>
void launch_sbvr_row_deq_kernel(
    __half* l_w,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int M, int N, int K,
    int use_shfl = 0,
    int device_id = 0)
{
    int blocks = cuda_prop_list[device_id].multiProcessorCount * BLOCK_PER_SM;
    dim3 threads = 32;

    if (use_shfl)
    {
        cuda_row_deq_mm_T<RIndexT, RNumSums><<<blocks, threads>>>(
            l_w,
            r_bvr, (RIndexT*)r_coeff_idx, r_coeff_cache,
            bias, out,
            M, N, K
        );
    }
    else
    {
        cuda_row_deq_wo_shfl_mm_T<RIndexT, RNumSums><<<blocks, threads>>>(
            l_w,
            r_bvr, (RIndexT*)r_coeff_idx, r_coeff_cache,
            bias, out,
            M, N, K
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

template <typename RIndexT>
void launch_sbvr_row_deq_kernel_wrapper(
    __half* l_w,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int M, int N, int K,
    int r_num_sums,
    int use_shfl = 0,
    int device_id = 0)
{
    RDKernelLaunchFN kernel_list[] = {
        launch_sbvr_row_deq_kernel<RIndexT, 2>,
        launch_sbvr_row_deq_kernel<RIndexT, 4>,
        launch_sbvr_row_deq_kernel<RIndexT, 6>,
        launch_sbvr_row_deq_kernel<RIndexT, 8>,
        launch_sbvr_row_deq_kernel<RIndexT, 10>
    };

    int kernel_idx = (r_num_sums - 2) / 2;
    kernel_list[kernel_idx](
        l_w,
        r_bvr, r_coeff_idx, r_coeff_cache,
        bias,
        out,
        M, N, K,
        use_shfl,
        device_id);
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
    int blocks = cuda_prop_list[device_id].multiProcessorCount * BLOCK_PER_SM;
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

template <typename LIndexT, typename RIndexT, int LNumSums, int RNumSums,
            int TileM, int TileN>
void launch_tMxtN_sbvr_kernel(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int device_id = 0)
{
    int blocks = cuda_prop_list[device_id].multiProcessorCount * BLOCK_PER_SM;
    dim3 threads = {BLOCK_TILE_SIZE / TileM, 
                    BLOCK_TILE_SIZE / TileN, 1};

    // std::cout << "Launching " << TileM << "x" << TileN << " SBVR kernel <" 
    //           << typeid(LIndexT).name() << ", " 
    //           << typeid(RIndexT).name() << ", "
    //           << "l_num_sums: " << LNumSums << ", "
    //           << "r_num_sums: " << RNumSums << ", "
    //           << "M: " << M << ", "
    //           << "N: " << N << ", "
    //           << "K: " << K << ", "
    //           << "blocks: " << blocks << ", " 
    //           << "threads: (" << threads.x << ", " 
    //           << threads.y << ", " << threads.z << ")" << std::endl;

    cuda_tMxtN_sbvr_mm_T<LIndexT, RIndexT, LNumSums, RNumSums, 
        TileM, TileN> <<<blocks, threads>>>(
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
        // <LIndexT, RIndexT, LNumSums, RNumSums, TileM, TileN>
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

template <typename LIndexT, typename RIndexT, int LNumSums, int RNumSums,
            int TileN>
void launch_1xtN_sbvr_kernel(
    uint32_t* l_bvr, void* l_coeff_idx, __half* l_coeff_cache,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias, __half* out,
    int M, int N, int K,
    int device_id = 0)
{
    // int blocks = 1;
    int blocks = cuda_prop_list[device_id].multiProcessorCount * BLOCK_PER_SM;
    dim3 threads = {TileN, THREAD_PER_WARP / TileN, WARP_PER_BLOCK};

    // std::cout << "Launching " << 1 << "x" << TileN << " SBVR kernel <" 
    //           << typeid(LIndexT).name() << ", " 
    //           << typeid(RIndexT).name() << ", "
    //           << "l_num_sums: " << LNumSums << ", "
    //           << "r_num_sums: " << RNumSums << ", "
    //           << "M: " << M << ", "
    //           << "N: " << N << ", "
    //           << "K: " << K << ", "
    //           << "blocks: " << blocks << ", " 
    //           << "threads: (" << threads.x << ", " 
    //           << threads.y << ", " << threads.z << ")" << std::endl;

    cuda_1xtN_sbvr_mm_T<LIndexT, RIndexT, LNumSums, RNumSums, 
        TileN> <<<blocks, threads>>>(
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
        // <LIndexT, RIndexT, LNumSums, RNumSums, TileN>
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 2, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 4, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 6, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 8, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 2, 10, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 2, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 4, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 6, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 8, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 4, 10, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 2, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 4, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 6, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 8, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 6, 10, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 2, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 4, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 6, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 8, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 8, 10, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 2, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 4, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 6, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 8, _1xtN_tN>,
        launch_1xtN_sbvr_kernel<LIndexT, RIndexT, 10, 10, _1xtN_tN>,
    };
    int kernel_idx = (l_num_sums - 2)/2 * 5 + (r_num_sums - 2)/2;
    kernel_list[kernel_idx](
           l_bvr, l_coeff_idx, l_coeff_cache,
           r_bvr, r_coeff_idx, r_coeff_cache,
           bias, out,
           M, N, K,
           device_id);
}

void launch_cuda_sbvr_row_deq_mm_T(
    __half* l_w,
    uint32_t* r_bvr, void* r_coeff_idx, __half* r_coeff_cache,
    __half* bias,
    __half* out,
    int M, int N, int K,
    int r_num_sums,
    int r_cache_size,
    int use_shfl = 0,
    int device_id = 0)
{
    const bool use_r_uint8 = (r_cache_size <= 256);
    if (use_r_uint8)
    {
        launch_sbvr_row_deq_kernel_wrapper<uint8_t>(
            l_w,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias,
            out,
            M, N, K,
            r_num_sums,
            use_shfl,
            device_id);
    }
    else
    {
        launch_sbvr_row_deq_kernel_wrapper<uint16_t>(
            l_w,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias,
            out,
            M, N, K,
            r_num_sums,
            use_shfl,
            device_id);
    }
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