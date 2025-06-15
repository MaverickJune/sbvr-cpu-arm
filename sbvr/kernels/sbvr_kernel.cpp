#include <torch/extension.h>
#include <iostream>
#include <cstdint>
#include <assert.h>

#include <thread>
#include <arm_neon.h>

#define K_PER_BVR 32 // BVR size 256 and bvr_dtype is uint8_t, so 256/8 = 32
#define N_LANE 16 // vcntq_u8 will process 16 lanes at a time

#include "thread_pool.hpp"

ThreadPool& global_pool()
{
    static ThreadPool pool;
    return pool;
}

extern "C" void sbvr_init_pool(int num_threads)
{
    global_pool().init(num_threads);
}

extern "C" void sbvr_finalize_pool()
{
    global_pool().finalize();
}


typedef void (*KernelLaunchFn)(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx, __fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K);


template <int NUM_SUMS>
struct coeffs {
    __fp16 i[NUM_SUMS];
};

void sbvr_neon_test() {

    float32x4_t a = {1.0f, 2.0f, 3.0f, 4.0f};
    float32x4_t b = {10.0f, 20.0f, 30.0f, 40.0f};

    float32x4_t result = vaddq_f32(a, b);

    float out[4];
    vst1q_f32(out, result);
    std::cout << "NEON result: ";
    for (int i = 0; i < 4; ++i)
        std::cout << out[i] << " ";
    std::cout << std::endl;

    // Check __FP16 support
    __fp16 h = 1.0f; // Implicit conversion from float to __fp16
    __fp16 h2 = 2.0f; // Another implicit conversion
    float16x8_t h_vec = vdupq_n_f16(h);
    float16x8_t h_vec2 = vdupq_n_f16(h2);
    float16x8_t h_result = vfmaq_f16(h_vec, h_vec2, h_vec2); // h_vec + h_vec2
    float16_t h_out[4];
    vst1_f16(h_out, vget_low_f16(h_result)); // Store lower 4 elements
    std::cout << "NEON __fp16 result: ";
    for (int i = 0; i < 4; ++i)
        std::cout << h_out[i] << " ";
    std::cout << std::endl;

#if defined(__ARM_FP16_FORMAT_IEEE)
    std::cout << "__ARM_FP16_FORMAT_IEEE is defined (IEEE fp16 supported)\n";
#else
    std::cout << "__ARM_FP16_FORMAT_IEEE is NOT defined\n";
    fprintf(stderr, "Error: IEEE fp16 format not supported.\n");
    return;
#endif

}

inline float32x4_t make_f32x4(float a, float b, float c, float d) {
    float tmp[4] = {a, b, c, d};
    return vld1q_f32(tmp);
}


template<
    typename  LIndexT, typename  RIndexT,
    int       LNumSums, int      RNumSums>
inline void
simd_kernel_1x16_no_fml( const uint8_t* __restrict l_bvr,        // (K, LNumSums)
                  const LIndexT* __restrict l_coeff_idx,
                  const __fp16* __restrict l_coeff_cache,

                  const uint8_t* __restrict r_bvr,   // (K, RNumSums , N_LANE)
                  const RIndexT* __restrict r_coeff_idx,
                  const __fp16* __restrict r_coeff_cache,

                  const __fp16* __restrict bias_pack,     // (N_LANE)
                  __fp16*       __restrict out_pack,       // (N_LANE)
                  int K)      
{

    const int bvr_per_K = K / K_PER_BVR; //

    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);

    for(int bvr_idx = 0; bvr_idx < bvr_per_K; ++bvr_idx)
    {

        // If all bits of l_bvr and r_bvr are set, this may overflow.
        uint8x16_t popc_cache[LNumSums][RNumSums];
        for (int lp = 0; lp < LNumSums; ++lp)
            for (int rp = 0; rp < RNumSums; ++rp)
                popc_cache[lp][rp] = vdupq_n_u8(0);

        for (int k = 0; k < K_PER_BVR; ++k)
        {
            const int k_idx = bvr_idx * K_PER_BVR + k;

            for (int l_idx = 0; l_idx < LNumSums / 2; ++l_idx)    // (a0,a1)
            {
                uint8_t  l0_b = l_bvr[(k_idx*LNumSums + (l_idx*2    ))]; // single byte
                uint8_t  l1_b = l_bvr[(k_idx*LNumSums + (l_idx*2 +1))];

                uint8x16_t l0 = vdupq_n_u8(l0_b);
                uint8x16_t l1 = vdupq_n_u8(l1_b);

                /* --------------------------------------------
                2.  R 쪽 대응 두 장  →  16-lane vector 로드
                -------------------------------------------- */
                for (int r_idx = 0; r_idx < RNumSums / 2; ++r_idx) // (b0,b1)
                {
                    const uint8_t *r_base =
                        &r_bvr[(k_idx * RNumSums + (r_idx * 2)) * 16]; // 16-lane base

                    uint8x16_t r0 = vld1q_u8(r_base     );
                    uint8x16_t r1 = vld1q_u8(r_base + 16);

                    const int li0 = l_idx*2;
                    const int li1 = li0 + 1;
                    const int rj0 = r_idx*2;
                    const int rj1 = rj0 + 1;

                    popc_cache[li0][rj0] = vaddq_u8(popc_cache[li0][rj0], vcntq_u8( vandq_u8(l0, r0) ));
                    popc_cache[li0][rj1] = vaddq_u8(popc_cache[li0][rj1], vcntq_u8( vandq_u8(l0, r1) ));
                    popc_cache[li1][rj0] = vaddq_u8(popc_cache[li1][rj0], vcntq_u8( vandq_u8(l1, r0) ));
                    popc_cache[li1][rj1] = vaddq_u8(popc_cache[li1][rj1], vcntq_u8( vandq_u8(l1, r1) ));
                }
            }
        }

        // Now we should multiply the counts by the coefficients

        const int l_coeff_i = l_coeff_idx[bvr_idx];
        int r_coeff_i[N_LANE];
        for(int n = 0; n < N_LANE; n++) {
            r_coeff_i[n] = r_coeff_idx[n * bvr_per_K + bvr_idx];
        }

        const coeffs<LNumSums> l_coeffs = *(coeffs<LNumSums>*)(&l_coeff_cache[l_coeff_i * LNumSums]);
        
        coeffs<RNumSums> r_coeffs_set[N_LANE];
        for(int n = 0; n < N_LANE; n++) {
            r_coeffs_set[n] = *reinterpret_cast<const coeffs<RNumSums>*>(&r_coeff_cache[r_coeff_i[n] * RNumSums]);
        }

        for (int l = 0; l < LNumSums; ++l)
        {
            float32_t a = static_cast<float>(l_coeffs.i[l]);
            float32x4_t a_vec = vdupq_n_f32(a);

            for (int r = 0; r < RNumSums; ++r)
            {
                float32x4_t b0123, b4567, b89AB, bCDEF;

#define B_LOAD(lane) static_cast<float>(r_coeffs_set[lane].i[r])

                b0123 = make_f32x4(B_LOAD(0), B_LOAD(1), B_LOAD(2), B_LOAD(3));
                b4567 = make_f32x4(B_LOAD(4), B_LOAD(5), B_LOAD(6), B_LOAD(7));
                b89AB = make_f32x4(B_LOAD(8), B_LOAD(9), B_LOAD(10), B_LOAD(11));
                bCDEF = make_f32x4(B_LOAD(12), B_LOAD(13), B_LOAD(14), B_LOAD(15));
#undef  B_LOAD

                //  popc[l][r] 16lane → 4×uint32x4_t로 나눠 변환
                uint8x16_t pc_u8 = popc_cache[l][r];
                uint16x8_t pc16_low = vmovl_u8(vget_low_u8(pc_u8));   // lane 0‥7
                uint16x8_t pc16_hi  = vmovl_u8(vget_high_u8(pc_u8));  // 8‥15
                float32x4_t pc0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (pc16_low)));
                float32x4_t pc1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(pc16_low)));
                float32x4_t pc2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (pc16_hi )));
                float32x4_t pc3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(pc16_hi )));

                //  acc += (a*b) * pc
                acc0 = vfmaq_f32(acc0, vmulq_f32(a_vec, b0123), pc0);
                acc1 = vfmaq_f32(acc1, vmulq_f32(a_vec, b4567), pc1);
                acc2 = vfmaq_f32(acc2, vmulq_f32(a_vec, b89AB), pc2);
                acc3 = vfmaq_f32(acc3, vmulq_f32(a_vec, bCDEF), pc3);
            }
        }
    }
    // float -> __fp16 Conversion
    float16x8_t out_vec0 = vcombine_f16(
        vcvt_f16_f32(acc0),
        vcvt_f16_f32(acc1)
    );
    float16x8_t out_vec1 = vcombine_f16(
        vcvt_f16_f32(acc2),
        vcvt_f16_f32(acc3)
    );

    // if bias is provided, add it
    if (bias_pack) {
        float16x8_t bias_vec_0 = vld1q_f16(bias_pack);
        float16x8_t bias_vec_1 = vld1q_f16(bias_pack + 8);
        out_vec0 = vaddq_f16(out_vec0, bias_vec_0);
        out_vec1 = vaddq_f16(out_vec1, bias_vec_1);
    }
    
    vst1q_f16(out_pack, out_vec0);
    vst1q_f16(out_pack + 8, out_vec1);

}

template<
    typename  LIndexT, typename  RIndexT,
    int       LNumSums, int      RNumSums>
inline void
simd_kernel_1x16_popc_first( const uint8_t* __restrict l_bvr,        // (K, LNumSums)
                  const LIndexT* __restrict l_coeff_idx,
                  const __fp16* __restrict l_coeff_cache,

                  const uint8_t* __restrict r_bvr,   // (K, RNumSums , N_LANE)
                  const RIndexT* __restrict r_coeff_idx,
                  const __fp16* __restrict r_coeff_cache,

                  const __fp16* __restrict bias_pack,     // (N_LANE)
                  __fp16*       __restrict out_pack,       // (N_LANE)
                  int K)      
{

    const int bvr_per_K = K / K_PER_BVR; //

    // If all bits of l_bvr and r_bvr are set, this may overflow.
    alignas(16) uint8x16_t popc_cache[LNumSums][RNumSums];

    // Now we should multiply the counts by the coefficients
    alignas(16) __fp16 lane_tile[ RNumSums ][ N_LANE ];

    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);

    for(int bvr_idx = 0; bvr_idx < bvr_per_K; ++bvr_idx)
    {

        const int l_coeff_i = l_coeff_idx[bvr_idx];

        for(int n = 0; n < N_LANE; n++) {
            const __fp16* src = r_coeff_cache + r_coeff_idx[n * bvr_per_K + bvr_idx] * RNumSums;
            for (int r = 0; r < RNumSums; ++r)
                lane_tile[r][n] = src[r];
        }

        memset(popc_cache, 0, sizeof popc_cache);


        for (int k = 0; k < K_PER_BVR; ++k)
        {
            const int k_idx = bvr_idx * K_PER_BVR + k;

            for (int l_idx = 0; l_idx < LNumSums / 2; ++l_idx)    // (a0,a1)
            {
                uint8_t  l0_b = l_bvr[(k_idx*LNumSums + (l_idx*2    ))]; // single byte
                uint8_t  l1_b = l_bvr[(k_idx*LNumSums + (l_idx*2 +1))];

                uint8x16_t l0 = vdupq_n_u8(l0_b);
                uint8x16_t l1 = vdupq_n_u8(l1_b);

                for (int r_idx = 0; r_idx < RNumSums / 2; ++r_idx) // (b0,b1)
                {
                    const uint8_t *r_base =
                        &r_bvr[(k_idx * RNumSums + (r_idx * 2)) * 16]; // 16-lane base

                    uint8x16_t r0 = vld1q_u8(r_base     );
                    uint8x16_t r1 = vld1q_u8(r_base + 16);

                    const int li0 = l_idx*2;
                    const int li1 = li0 + 1;
                    const int rj0 = r_idx*2;
                    const int rj1 = rj0 + 1;

                    popc_cache[li0][rj0] = vaddq_u8(popc_cache[li0][rj0], vcntq_u8( vandq_u8(l0, r0) ));
                    popc_cache[li0][rj1] = vaddq_u8(popc_cache[li0][rj1], vcntq_u8( vandq_u8(l0, r1) ));
                    popc_cache[li1][rj0] = vaddq_u8(popc_cache[li1][rj0], vcntq_u8( vandq_u8(l1, r0) ));
                    popc_cache[li1][rj1] = vaddq_u8(popc_cache[li1][rj1], vcntq_u8( vandq_u8(l1, r1) ));
                }
            }
        }
             

        for (int l = 0; l < LNumSums; l += 2) {
            const __fp16 a0 = l_coeff_cache[l_coeff_i * LNumSums + l];
            const __fp16 a1 = l_coeff_cache[l_coeff_i * LNumSums + l + 1];
            float16x8_t a0v = vdupq_n_f16(a0);
            float16x8_t a1v = vdupq_n_f16(a1);

            for (int r = 0; r < RNumSums; ++r) {
                float16x8_t b0 = vld1q_f16(&lane_tile[r][0]);
                float16x8_t b1 = vld1q_f16(&lane_tile[r][8]);

                uint8x16_t pc_u8   = popc_cache[l][r];
                uint8x8_t  pc_low8 = vget_low_u8 (pc_u8);
                uint8x8_t  pc_hi8  = vget_high_u8(pc_u8);

                float16x8_t pc_lo = vcvtq_f16_u16(vmovl_u8(pc_low8));
                float16x8_t pc_hi = vcvtq_f16_u16(vmovl_u8(pc_hi8));

                /* ---- a0 path ------------------------------------------------ */
                float16x8_t ab0 = vmulq_f16(a0v, b0);
                acc0 = vfmlalq_low_f16 (acc0, ab0, pc_lo);
                acc1 = vfmlalq_high_f16(acc1, ab0, pc_lo);

                float16x8_t ab1 = vmulq_f16(a0v, b1);
                acc2 = vfmlalq_low_f16 (acc2, ab1, pc_hi);
                acc3 = vfmlalq_high_f16(acc3, ab1, pc_hi);

                /* ---- a1 path ------------------------------------------------ */
                pc_u8   = popc_cache[l + 1][r];
                pc_low8 = vget_low_u8 (pc_u8);
                pc_hi8  = vget_high_u8(pc_u8);

                pc_lo = vcvtq_f16_u16(vmovl_u8(pc_low8));
                pc_hi = vcvtq_f16_u16(vmovl_u8(pc_hi8));

                ab0 = vmulq_f16(a1v, b0);
                acc0 = vfmlalq_low_f16 (acc0, ab0, pc_lo);
                acc1 = vfmlalq_high_f16(acc1, ab0, pc_lo);

                ab1 = vmulq_f16(a1v, b1);
                acc2 = vfmlalq_low_f16 (acc2, ab1, pc_hi);
                acc3 = vfmlalq_high_f16(acc3, ab1, pc_hi);
            }
        }

    }
    // float -> __fp16 Conversion
    float16x8_t out_vec0 = vcombine_f16(
        vcvt_f16_f32(acc0),
        vcvt_f16_f32(acc1)
    );
    float16x8_t out_vec1 = vcombine_f16(
        vcvt_f16_f32(acc2),
        vcvt_f16_f32(acc3)
    );

    // if bias is provided, add it
    if (bias_pack) {
        float16x8_t bias_vec_0 = vld1q_f16(bias_pack);
        float16x8_t bias_vec_1 = vld1q_f16(bias_pack + 8);
        out_vec0 = vaddq_f16(out_vec0, bias_vec_0);
        out_vec1 = vaddq_f16(out_vec1, bias_vec_1);
    }
    
    vst1q_f16(out_pack, out_vec0);
    vst1q_f16(out_pack + 8, out_vec1);

}



template<
    typename  LIndexT, typename  RIndexT,
    int       LNumSums, int      RNumSums>
struct WorkerArg {
    uint8_t* l_bvr;                // (K, LNumSums)
    void* l_coeff_idx;             // (K / K_PER_BVR, M)
    __fp16* l_coeff_cache;         // (l_cache_size, LNumSums)

    uint8_t* r_bvr;                // (N / N_LANE, K, RNumSums, N_LANE)
    void* r_coeff_idx;             // (N, K / K_PER_BVR)
    __fp16* r_coeff_cache;         // (r_cache_size, RNumSums)

    __fp16* bias;                  // (N_LANE)
    __fp16* out;                   // (N / N_LANE, N_LANE)

    int K;                          // Number of features
    int N;                          // Total number of output columns
    int n_begin;                   // Start column index for this worker
    int n_end;                     // End column index for this worker
};

template<typename LIndexT, typename RIndexT,
         int LNumSums,   int RNumSums>
inline void worker_fn(const WorkerArg<LIndexT, RIndexT, LNumSums, RNumSums>* __restrict a)
{
    constexpr int LANES          = N_LANE;              // 16
    const int     bvr_per_K      = a->K / K_PER_BVR;

    const int stride_r_pack      = a->K * RNumSums * LANES;     // bytes
    const int stride_coeff_pack  = LANES * bvr_per_K;           // elements
    const int stride_out_pack    = LANES;                       // __fp16 elements


    const int pack_begin         = a->n_begin / LANES;    // /16
    const int pack_end           = a->n_end   / LANES;    // /16

    const uint8_t*       r_pack      = a->r_bvr        + pack_begin * stride_r_pack;
    const RIndexT*       r_coeff_idx = (const RIndexT*)a->r_coeff_idx + pack_begin * stride_coeff_pack;
    __fp16*              out_pack    = a->out          + pack_begin * stride_out_pack;
    const __fp16*        bias_base   = a->bias;        // may be nullptr

    for (int pk = pack_begin; pk < pack_end; ++pk)
    {
        const __fp16* bias_pack = bias_base ? bias_base + pk * LANES : nullptr;

        simd_kernel_1x16_popc_first<LIndexT, RIndexT, LNumSums, RNumSums>(
            a->l_bvr,
            (const LIndexT*)a->l_coeff_idx,
            a->l_coeff_cache,
            r_pack,
            r_coeff_idx,
            a->r_coeff_cache,
            bias_pack,
            out_pack,
            a->K);

        r_pack      += stride_r_pack;
        r_coeff_idx += stride_coeff_pack;
        out_pack    += stride_out_pack;
    }
}


//------------------------------------------------------------
// 3. 상위 래퍼 : pthread 분할 후 워커 호출
//------------------------------------------------------------
template<
    typename  LIndexT, typename  RIndexT,
    int       LNumSums, int      RNumSums>
void sbvr_mm_cpu_1xN(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx, __fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K)
{
    const int num_threads = global_pool().num_threads();

    const int chunk = (N + num_threads - 1) / num_threads;
    std::vector<WorkerArg<LIndexT,RIndexT,LNumSums,RNumSums>> args;

    int n_tasks = 0;
    for (int t = 0; t < num_threads; ++t) {
        int n0 =  t * chunk;
        int n1 = std::min(n0 + chunk, N);
        if (n0 >= n1) break;
        args.push_back({ l_bvr, l_coeff_idx, l_coeff_cache,
                         r_bvr, r_coeff_idx, r_coeff_cache,
                         bias,  out, K, N, n0, n1 });
        ++n_tasks;
    }

    global_pool().parallel_for(n_tasks, [&](int task_id){
        worker_fn<LIndexT,RIndexT,LNumSums,RNumSums>(&args[task_id]);
    });
}

template <typename LIndexT, typename RIndexT>
void launch_cpu_sbvr_kernel_wrapper(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx,__fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size)
{
    KernelLaunchFn kernel_list[] = {
        // <LIndexT, RIndexT, LNumSums, RNumSums>
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 2, 2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 2, 4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 2, 6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 2, 8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 2, 10>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 4, 2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 4, 4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 4, 6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 4, 8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 4, 10>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 8, 2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 8, 4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 8, 6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 8, 8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 8, 10>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10, 2>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10, 4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10, 6>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10, 8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 10, 10>,
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
        M, N, K);
        
}

void launch_cpu_sbvr_mm_1xN(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx,__fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size)
{
    
    const bool use_l_uint8 = (l_cache_size <= 256);
    const bool use_r_uint8 = (r_cache_size <= 256);

    // Each num_sums must be 4 or 8
    const bool supported_num_sums = 
        (l_num_sums % 2 == 0 && r_num_sums % 2 == 0) &&
        (l_num_sums >= 2 && l_num_sums <= 10) &&
        (r_num_sums >= 2 && r_num_sums <= 10);

    
    if (supported_num_sums && use_l_uint8 && use_r_uint8) {
        // Use uint8_t for both left and right BVRs
        if(use_l_uint8 && use_r_uint8) {
            launch_cpu_sbvr_kernel_wrapper<uint8_t, uint8_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                l_cache_size, r_cache_size);
        } else if (use_l_uint8 && !use_r_uint8) {
            // Use uint8_t for left BVR and uint16_t for right BVR
            launch_cpu_sbvr_kernel_wrapper<uint8_t, uint16_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                l_cache_size, r_cache_size);
        } else if (!use_l_uint8 && use_r_uint8) {
            // Use uint16_t for left BVR and uint8_t for right BVR
            launch_cpu_sbvr_kernel_wrapper<uint16_t, uint8_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                l_cache_size, r_cache_size);
        } else {
            // Use uint16_t for both left and right BVRs
            launch_cpu_sbvr_kernel_wrapper<uint16_t, uint16_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                l_cache_size, r_cache_size);
        }
    } else {
        // To-Do : Naive implementation for other cases
        std::cerr << "Unsupported configuration: "
                  << "l_num_sums=" << l_num_sums
                  << ", r_num_sums=" << r_num_sums
                  << ", l_cache_size=" << l_cache_size
                  << ", r_cache_size=" << r_cache_size
                  << std::endl;
    }
}

torch::Tensor sbvr_cpu_mm_T(
                torch::Tensor l_bvr,  // (K, M, LNumSums)
                torch::Tensor l_coeff_idx, // (K / K_PER_BVR, M)
                torch::Tensor l_coeff_cache, // (l_cache_size, LNumSums)
                torch::Tensor r_bvr, // (N / N_LANE, K, RNumSums, N_LANE)
                torch::Tensor r_coeff_idx, // (N, K / K_PER_BVR)
                torch::Tensor r_coeff_cache, // (r_cache_size, RNumSums)
                torch::Tensor bias
            )
{
    const int M = l_bvr.size(1);
    const int N = r_bvr.size(0) * r_bvr.size(3); // N_LANE
    const int K = l_bvr.size(0); 
    const int l_num_sums = l_bvr.size(2);
    const int r_num_sums = r_bvr.size(2);
    const int l_cache_size = l_coeff_cache.size(0);
    const int r_cache_size = r_coeff_cache.size(0);
    // assert (l_bvr.size(0) == r_bvr.size(1));

    auto out = torch::empty({M, N},
                         torch::dtype(torch::kFloat16).device(l_bvr.device()));
    __fp16* bias_ptr = nullptr;
    if (bias.size(0) == N)
        bias_ptr = reinterpret_cast<__fp16*>(bias.data_ptr<at::Half>());
    
    // Call the dispatch kernel
    launch_cpu_sbvr_mm_1xN(
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_sbvr_neon_test", &sbvr_neon_test,
            "Simple NEON vector addition test");
    m.def("_sbvr_cpu_mm_T", &sbvr_cpu_mm_T,
            "SBVR matrix multiplication on CPU with NEON support",
            py::arg("l_bvr"), py::arg("l_coeff_idx"), py::arg("l_coeff_cache"),
            py::arg("r_bvr"), py::arg("r_coeff_idx"), py::arg("r_coeff_cache"),
            py::arg("bias") = torch::Tensor());

    m.def("_sbvr_init_pool", &sbvr_init_pool, py::arg("num_threads"));
    m.def("_sbvr_finalize_pool", &sbvr_finalize_pool);
}