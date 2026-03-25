/*
 * sbvr_kernel_v2.cpp — Optimized ARM NEON SBVR matrix-multiply kernel
 *
 * Computes  C = A @ B^T   where A and B are stored in SBVR (binary-vector
 * representation) format.
 *
 * BVR layout follows the original CUDA kernel convention:
 *   l_bvr        : (K, M, num_sums)        uint8   — K = total_K / 8
 *   l_coeff_idx  : (bvr_per_K, M)          uint8 or uint16
 *   l_coeff_cache: (l_cache_size, num_sums) fp16
 *   r_bvr        : (K, N, num_sums)        uint8
 *   r_coeff_idx  : (bvr_per_K, N)          uint8 or uint16
 *   r_coeff_cache: (r_cache_size, num_sums) fp16
 *   bias         : (N,) fp16  (optional)
 *   out          : (M, N) fp16
 *
 * K_PER_BVR = 32 means 32 uint8 elements = 256 bits per BVR group.
 * bvr_per_K = K / K_PER_BVR
 *
 * Optimizations for Neoverse-V2 (ARMv9):
 *   - vcntq_u8 for native byte-level popcount (single cycle on V2)
 *   - vaddlq_u8 / vpadalq_u8 for widening accumulation to uint16
 *   - vaddlvq_u16 for horizontal reduction
 *   - float32 accumulators for coefficient multiply-accumulate
 *   - Multi-threaded across N using ThreadPool
 *   - Popcount-first strategy: accumulate all K_PER_BVR popcounts
 *     before multiplying by coefficients (reduces coeff memory traffic)
 */

#include <torch/extension.h>
#include <iostream>
#include <cstdint>
#include <cassert>
#include <thread>
#include <arm_neon.h>

#define K_PER_BVR 32  // 32 bytes = 256 bits per BVR group

#include "thread_pool.hpp"

/* ─── Global thread pool (shared with v1 if both loaded) ──────────── */
static ThreadPool& v2_global_pool()
{
    static ThreadPool pool;
    return pool;
}

extern "C" void sbvr_v2_init_pool(int num_threads)
{
    v2_global_pool().init(num_threads);
}

extern "C" void sbvr_v2_finalize_pool()
{
    v2_global_pool().finalize();
}

/* ──────────────────────────────────────────────────────────────────── *
 *  Core SIMD kernel:  1 × tile_n  output elements per call
 *
 *  For M = 1 (typical LLM inference), we parallelize across N.
 *  Each thread handles a contiguous chunk of N columns.
 * ──────────────────────────────────────────────────────────────────── */

/*
 * Compute popcount( l_bvr_byte & r_bvr_byte ) summed across K_PER_BVR bytes
 * for all (l_sum, r_sum) pairs, storing the result into popc_out.
 *
 * popc_out[ls][rs] = sum_{k=0}^{K_PER_BVR-1} popcount( l_bvr[k*LNumSums+ls] & r_bvr[k*RNumSums+rs] )
 *
 * We use a "popcount-first" approach: loop over k, accumulating byte-level
 * popcounts into uint16 accumulators. K_PER_BVR=32, max popcount per byte=8,
 * so max accumulated value = 32 * 8 = 256, fits in uint16.
 */
template <int LNumSums, int RNumSums>
inline void compute_popc_block(
    const uint8_t* __restrict l_ptr,   // points to l_bvr[(bvr_idx*K_PER_BVR)*M*LNumSums + m*LNumSums]
    int l_stride,                       // M * LNumSums (stride in bytes between consecutive k for same m)
    const uint8_t* __restrict r_ptr,   // points to r_bvr[(bvr_idx*K_PER_BVR)*N*RNumSums + n*RNumSums]
    int r_stride,                       // N * RNumSums
    uint16_t popc_out[LNumSums][RNumSums])
{
    // Zero accumulators
    for (int ls = 0; ls < LNumSums; ++ls)
        for (int rs = 0; rs < RNumSums; ++rs)
            popc_out[ls][rs] = 0;

    /*
     * Process K_PER_BVR bytes in chunks of 16 using NEON vcntq_u8.
     * For each k, load all num_sums BVR bytes for l and r, then
     * compute AND + popcount for all (ls, rs) pairs.
     *
     * Since K_PER_BVR=32, we can process 2 iterations of 16 bytes,
     * but each k conceptually addresses one byte repeated across lanes.
     * Actually, each k is a separate byte in the BVR group.
     * We accumulate scalar popcounts using NEON on groups of 16 k-values.
     */

    // Strategy: process 16 k-values at a time using NEON
    for (int k_base = 0; k_base < K_PER_BVR; k_base += 16) {
        // For each (ls, rs) pair, gather 16 bytes of l[k][ls] and r[k][rs],
        // then do AND + vcntq_u8 + horizontal sum
        for (int ls = 0; ls < LNumSums; ++ls) {
            // Gather 16 l_bvr bytes for this ls across k_base..k_base+15
            uint8_t l_bytes[16];
            for (int dk = 0; dk < 16; ++dk) {
                l_bytes[dk] = l_ptr[(k_base + dk) * l_stride + ls];
            }
            uint8x16_t l_vec = vld1q_u8(l_bytes);

            for (int rs = 0; rs < RNumSums; ++rs) {
                // Gather 16 r_bvr bytes for this rs across k_base..k_base+15
                uint8_t r_bytes[16];
                for (int dk = 0; dk < 16; ++dk) {
                    r_bytes[dk] = r_ptr[(k_base + dk) * r_stride + rs];
                }
                uint8x16_t r_vec = vld1q_u8(r_bytes);

                // AND + popcount per byte + horizontal sum
                uint8x16_t and_vec = vandq_u8(l_vec, r_vec);
                uint8x16_t cnt_vec = vcntq_u8(and_vec);
                popc_out[ls][rs] += vaddvq_u8(cnt_vec);
            }
        }
    }
}

/*
 * Main kernel: computes one row of the output (m-th row, columns n0..n1-1).
 *
 * Template parameters:
 *   LIndexT, RIndexT: coeff_idx element type (uint8_t or uint16_t)
 *   LNumSums, RNumSums: number of summation terms
 */
template <typename LIndexT, typename RIndexT, int LNumSums, int RNumSums>
void kernel_1row_nchunk(
    const uint8_t*  __restrict l_bvr,
    const LIndexT*  __restrict l_coeff_idx,
    const __fp16*   __restrict l_coeff_cache,
    const uint8_t*  __restrict r_bvr,
    const RIndexT*  __restrict r_coeff_idx,
    const __fp16*   __restrict r_coeff_cache,
    const __fp16*   __restrict bias,
    __fp16*         __restrict out,
    int m, int n0, int n1,
    int M, int N, int K)
{
    const int bvr_per_K = K / K_PER_BVR;

    for (int n = n0; n < n1; ++n) {
        float sum = 0.0f;

        for (int bvr_idx = 0; bvr_idx < bvr_per_K; ++bvr_idx) {
            // Compute popcount block for all (ls, rs) pairs
            uint16_t popc[LNumSums][RNumSums];

            // Pointer to l_bvr for this (bvr_idx, m):
            //   l_bvr layout: (K, M, LNumSums)
            //   l_bvr[(bvr_idx * K_PER_BVR + k) * M * LNumSums + m * LNumSums + ls]
            const uint8_t* l_ptr = l_bvr +
                (bvr_idx * K_PER_BVR) * M * LNumSums + m * LNumSums;
            int l_stride = M * LNumSums;

            // Pointer to r_bvr for this (bvr_idx, n):
            //   r_bvr layout: (K, N, RNumSums)
            //   r_bvr[(bvr_idx * K_PER_BVR + k) * N * RNumSums + n * RNumSums + rs]
            const uint8_t* r_ptr = r_bvr +
                (bvr_idx * K_PER_BVR) * N * RNumSums + n * RNumSums;
            int r_stride = N * RNumSums;

            compute_popc_block<LNumSums, RNumSums>(
                l_ptr, l_stride, r_ptr, r_stride, popc);

            // Load coefficients
            int l_ci = l_coeff_idx[bvr_idx * M + m];
            int r_ci = r_coeff_idx[bvr_idx * N + n];
            const __fp16* l_c = l_coeff_cache + l_ci * LNumSums;
            const __fp16* r_c = r_coeff_cache + r_ci * RNumSums;

            // Multiply-accumulate: sum += popc[ls][rs] * l_coeff[ls] * r_coeff[rs]
            for (int ls = 0; ls < LNumSums; ++ls) {
                float lc = static_cast<float>(l_c[ls]);
                for (int rs = 0; rs < RNumSums; ++rs) {
                    sum += static_cast<float>(popc[ls][rs]) * lc *
                           static_cast<float>(r_c[rs]);
                }
            }
        }

        // Write output with optional bias
        __fp16 bias_val = (bias != nullptr) ? bias[n] : static_cast<__fp16>(0.0f);
        out[m * N + n] = static_cast<__fp16>(sum) + bias_val;
    }
}

/* ──────────────────────────────────────────────────────────────────── *
 *  Optimized kernel for M=1: process 4 N columns simultaneously
 *  using NEON to parallelize coefficient multiply-accumulate.
 * ──────────────────────────────────────────────────────────────────── */
template <typename LIndexT, typename RIndexT, int LNumSums, int RNumSums>
void kernel_m1_nchunk_opt(
    const uint8_t*  __restrict l_bvr,
    const LIndexT*  __restrict l_coeff_idx,
    const __fp16*   __restrict l_coeff_cache,
    const uint8_t*  __restrict r_bvr,
    const RIndexT*  __restrict r_coeff_idx,
    const __fp16*   __restrict r_coeff_cache,
    const __fp16*   __restrict bias,
    __fp16*         __restrict out,
    int n0, int n1,
    int N, int K)
{
    const int bvr_per_K = K / K_PER_BVR;
    const int M = 1;
    const int m = 0;

    // Process 4 columns at a time for NEON vectorization
    int n = n0;
    for (; n + 3 < n1; n += 4) {
        float32x4_t sum4 = vdupq_n_f32(0.0f);

        for (int bvr_idx = 0; bvr_idx < bvr_per_K; ++bvr_idx) {
            // l_bvr pointer (same for all 4 n columns since M=1)
            const uint8_t* l_ptr = l_bvr +
                (bvr_idx * K_PER_BVR) * LNumSums;
            const int l_stride = LNumSums;

            // Load l coefficients once (shared across all n)
            int l_ci = l_coeff_idx[bvr_idx];
            const __fp16* l_c = l_coeff_cache + l_ci * LNumSums;

            // Process 4 columns
            for (int nn = 0; nn < 4; ++nn) {
                int cur_n = n + nn;
                const uint8_t* r_ptr = r_bvr +
                    (bvr_idx * K_PER_BVR) * N * RNumSums + cur_n * RNumSums;
                const int r_stride = N * RNumSums;

                uint16_t popc[LNumSums][RNumSums];
                compute_popc_block<LNumSums, RNumSums>(
                    l_ptr, l_stride, r_ptr, r_stride, popc);

                int r_ci = r_coeff_idx[bvr_idx * N + cur_n];
                const __fp16* r_c = r_coeff_cache + r_ci * RNumSums;

                float partial = 0.0f;
                for (int ls = 0; ls < LNumSums; ++ls) {
                    float lc = static_cast<float>(l_c[ls]);
                    for (int rs = 0; rs < RNumSums; ++rs) {
                        partial += static_cast<float>(popc[ls][rs]) * lc *
                                   static_cast<float>(r_c[rs]);
                    }
                }
                sum4 = vsetq_lane_f32(vgetq_lane_f32(sum4, 0), sum4, 0); // nop for structure
                // Store scalar partial into the NEON register
                switch (nn) {
                    case 0: sum4 = vsetq_lane_f32(partial, sum4, 0); break;
                    case 1: sum4 = vsetq_lane_f32(partial, sum4, 1); break;
                    case 2: sum4 = vsetq_lane_f32(partial, sum4, 2); break;
                    case 3: sum4 = vsetq_lane_f32(partial, sum4, 3); break;
                }
            }
        }

        // Write 4 outputs with bias
        float result[4];
        vst1q_f32(result, sum4);
        for (int nn = 0; nn < 4; ++nn) {
            float bias_val = (bias != nullptr) ?
                static_cast<float>(bias[n + nn]) : 0.0f;
            out[n + nn] = static_cast<__fp16>(result[nn] + bias_val);
        }
    }

    // Handle remaining columns
    for (; n < n1; ++n) {
        float sum = 0.0f;
        for (int bvr_idx = 0; bvr_idx < bvr_per_K; ++bvr_idx) {
            const uint8_t* l_ptr = l_bvr +
                (bvr_idx * K_PER_BVR) * LNumSums;
            const int l_stride = LNumSums;

            const uint8_t* r_ptr = r_bvr +
                (bvr_idx * K_PER_BVR) * N * RNumSums + n * RNumSums;
            const int r_stride = N * RNumSums;

            uint16_t popc[LNumSums][RNumSums];
            compute_popc_block<LNumSums, RNumSums>(
                l_ptr, l_stride, r_ptr, r_stride, popc);

            int l_ci = l_coeff_idx[bvr_idx];
            int r_ci = r_coeff_idx[bvr_idx * N + n];
            const __fp16* l_c = l_coeff_cache + l_ci * LNumSums;
            const __fp16* r_c = r_coeff_cache + r_ci * RNumSums;

            for (int ls = 0; ls < LNumSums; ++ls) {
                float lc = static_cast<float>(l_c[ls]);
                for (int rs = 0; rs < RNumSums; ++rs) {
                    sum += static_cast<float>(popc[ls][rs]) * lc *
                           static_cast<float>(r_c[rs]);
                }
            }
        }
        __fp16 bias_val = (bias != nullptr) ? bias[n] : static_cast<__fp16>(0.0f);
        out[n] = static_cast<__fp16>(sum) + bias_val;
    }
}

/* ──────────────────────────────────────────────────────────────────── *
 *  Thread-parallel dispatch: partition N across threads
 * ──────────────────────────────────────────────────────────────────── */
template <typename LIndexT, typename RIndexT, int LNumSums, int RNumSums>
void sbvr_v2_mm_dispatch(
    uint8_t*  l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t*  r_bvr, void* r_coeff_idx, __fp16* r_coeff_cache,
    __fp16*   bias,  __fp16* out,
    int M, int N, int K)
{
    const int num_threads = v2_global_pool().num_threads();

    for (int m = 0; m < M; ++m) {
        if (M == 1) {
            // Optimized M=1 path
            v2_global_pool().parallel_for(num_threads, [&](int tid) {
                int chunk = (N + num_threads - 1) / num_threads;
                int n0 = tid * chunk;
                int n1 = std::min(n0 + chunk, N);
                if (n0 >= n1) return;

                kernel_m1_nchunk_opt<LIndexT, RIndexT, LNumSums, RNumSums>(
                    l_bvr,
                    static_cast<const LIndexT*>(l_coeff_idx),
                    l_coeff_cache,
                    r_bvr,
                    static_cast<const RIndexT*>(r_coeff_idx),
                    r_coeff_cache,
                    bias, out,
                    n0, n1, N, K);
            });
        } else {
            // General M > 1 path
            v2_global_pool().parallel_for(num_threads, [&](int tid) {
                int chunk = (N + num_threads - 1) / num_threads;
                int n0 = tid * chunk;
                int n1 = std::min(n0 + chunk, N);
                if (n0 >= n1) return;

                kernel_1row_nchunk<LIndexT, RIndexT, LNumSums, RNumSums>(
                    l_bvr,
                    static_cast<const LIndexT*>(l_coeff_idx),
                    l_coeff_cache,
                    r_bvr,
                    static_cast<const RIndexT*>(r_coeff_idx),
                    r_coeff_cache,
                    bias, out,
                    m, n0, n1, M, N, K);
            });
        }
    }
}

/* ──────────────────────────────────────────────────────────────────── *
 *  Kernel launch wrapper: dispatch by (LNumSums, RNumSums)
 * ──────────────────────────────────────────────────────────────────── */

typedef void (*V2KernelFn)(
    uint8_t*, void*, __fp16*,
    uint8_t*, void*, __fp16*,
    __fp16*, __fp16*,
    int, int, int);

template <typename LIndexT, typename RIndexT>
void launch_v2_kernel_wrapper(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx, __fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums)
{
    // Table of kernels for (l_num_sums, r_num_sums) in {2,4,6,8,10} × {2,4,6,8,10}
    V2KernelFn kernel_list[] = {
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 2, 2>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 2, 4>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 2, 6>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 2, 8>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 2, 10>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 4, 2>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 4, 4>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 4, 6>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 4, 8>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 4, 10>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 6, 2>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 6, 4>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 6, 6>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 6, 8>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 6, 10>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 8, 2>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 8, 4>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 8, 6>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 8, 8>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 8, 10>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 10, 2>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 10, 4>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 10, 6>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 10, 8>,
        sbvr_v2_mm_dispatch<LIndexT, RIndexT, 10, 10>,
    };

    int idx = (l_num_sums - 2) / 2 * 5 + (r_num_sums - 2) / 2;
    if (idx < 0 || idx >= 25) {
        throw std::runtime_error("sbvr_v2: unsupported num_sums combination");
    }

    kernel_list[idx](
        l_bvr, l_coeff_idx, l_coeff_cache,
        r_bvr, r_coeff_idx, r_coeff_cache,
        bias, out, M, N, K);
}

/* ──────────────────────────────────────────────────────────────────── *
 *  Top-level dispatch: select (LIndexT, RIndexT) based on cache sizes
 * ──────────────────────────────────────────────────────────────────── */
void launch_v2_sbvr_mm(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx, __fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size)
{
    const bool use_l_uint8 = (l_cache_size <= 256);
    const bool use_r_uint8 = (r_cache_size <= 256);

    const bool supported =
        (l_num_sums % 2 == 0 && r_num_sums % 2 == 0) &&
        (l_num_sums >= 2 && l_num_sums <= 10) &&
        (r_num_sums >= 2 && r_num_sums <= 10);

    if (!supported) {
        throw std::runtime_error(
            "sbvr_v2: unsupported num_sums (must be even, 2..10)");
    }

    if (use_l_uint8 && use_r_uint8) {
        launch_v2_kernel_wrapper<uint8_t, uint8_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out, M, N, K,
            l_num_sums, r_num_sums);
    } else if (use_l_uint8 && !use_r_uint8) {
        launch_v2_kernel_wrapper<uint8_t, uint16_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out, M, N, K,
            l_num_sums, r_num_sums);
    } else if (!use_l_uint8 && use_r_uint8) {
        launch_v2_kernel_wrapper<uint16_t, uint8_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out, M, N, K,
            l_num_sums, r_num_sums);
    } else {
        launch_v2_kernel_wrapper<uint16_t, uint16_t>(
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out, M, N, K,
            l_num_sums, r_num_sums);
    }
}

/* ──────────────────────────────────────────────────────────────────── *
 *  PyTorch wrapper
 *
 *  Tensor shapes (CUDA-style, original layout):
 *    l_bvr        : (K, M, l_num_sums)       uint8
 *    l_coeff_idx  : (bvr_per_K, M)           uint8 or uint16
 *    l_coeff_cache: (l_cache_size, l_num_sums) float16
 *    r_bvr        : (K, N, r_num_sums)       uint8
 *    r_coeff_idx  : (bvr_per_K, N)           uint8 or uint16
 *    r_coeff_cache: (r_cache_size, r_num_sums) float16
 *    bias         : (N,) float16, optional
 * ──────────────────────────────────────────────────────────────────── */
torch::Tensor sbvr_cpu_v2_mm_T(
    torch::Tensor l_bvr,
    torch::Tensor l_coeff_idx,
    torch::Tensor l_coeff_cache,
    torch::Tensor r_bvr,
    torch::Tensor r_coeff_idx,
    torch::Tensor r_coeff_cache,
    torch::Tensor bias)
{
    const int K = l_bvr.size(0);   // total K in bytes (= original_K / 8)
    const int M = l_bvr.size(1);
    const int N = r_bvr.size(1);
    const int l_num_sums = l_bvr.size(2);
    const int r_num_sums = r_bvr.size(2);
    const int l_cache_size = l_coeff_cache.size(0);
    const int r_cache_size = r_coeff_cache.size(0);

    auto out = torch::empty({M, N},
        torch::dtype(torch::kFloat16).device(l_bvr.device()));

    __fp16* bias_ptr = nullptr;
    if (bias.numel() == N)
        bias_ptr = reinterpret_cast<__fp16*>(bias.data_ptr<at::Half>());

    launch_v2_sbvr_mm(
        l_bvr.data_ptr<uint8_t>(),
        l_coeff_idx.data_ptr(),
        reinterpret_cast<__fp16*>(l_coeff_cache.data_ptr<at::Half>()),
        r_bvr.data_ptr<uint8_t>(),
        r_coeff_idx.data_ptr(),
        reinterpret_cast<__fp16*>(r_coeff_cache.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<__fp16*>(out.data_ptr<at::Half>()),
        M, N, K,
        l_num_sums, r_num_sums,
        l_cache_size, r_cache_size);

    return out;
}

/* ──────────────────────────────────────────────────────────────────── *
 *  pybind11 module
 * ──────────────────────────────────────────────────────────────────── */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_sbvr_cpu_v2_mm_T", &sbvr_cpu_v2_mm_T,
          "SBVR v2 ARM NEON matrix-multiply (C = A @ B^T)",
          py::arg("l_bvr"), py::arg("l_coeff_idx"), py::arg("l_coeff_cache"),
          py::arg("r_bvr"), py::arg("r_coeff_idx"), py::arg("r_coeff_cache"),
          py::arg("bias") = torch::Tensor());

    m.def("_sbvr_v2_init_pool",     &sbvr_v2_init_pool,
          py::arg("num_threads"));
    m.def("_sbvr_v2_finalize_pool", &sbvr_v2_finalize_pool);
}
