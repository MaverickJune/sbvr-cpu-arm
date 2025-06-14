#include <torch/extension.h>
#include <iostream>
#include <cstdint>
#include <assert.h>

#include <thread>
#include <arm_neon.h>

#define K_PER_BVR 32 // BVR size 256 and bvr_dtype is uint8_t, so 256/8 = 32
#define N_LANE 16 // vcntq_u8 will process 16 lanes at a time

typedef void (*KernelLaunchFn)(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx, __fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K, int num_threads);


template <int NUM_SUMS>
struct coeffs;

template <>
struct coeffs<4> {
    __fp16 i[4];
};

template <>
struct coeffs<8> {
    __fp16 i[8];
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

// #if defined(__ARM_FEATURE_F16_SCALAR_ARITHMETIC)
//     std::cout << "__ARM_FEATURE_F16_SCALAR_ARITHMETIC is defined (scalar fp16 supported)\n";
// #else
//     std::cout << "__ARM_FEATURE_F16_SCALAR_ARITHMETIC is NOT defined\n";
// #endif
}

inline float32x4_t make_f32x4(float a, float b, float c, float d) {
    float tmp[4] = {a, b, c, d};
    return vld1q_f32(tmp);
}


//------------------------------------------------------------
// 1. SIMD 마이크로커널 (열 16개 묶음 기준) – 구현은 직접 작성
//------------------------------------------------------------
template<
    typename  LIndexT, typename  RIndexT,
    int       LNumSums, int      RNumSums>
inline void
simd_kernel_1x16( const uint8_t* __restrict l_bvr,        // (K, LNumSums)
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
        alignas(16) __fp16 lane_tile[ RNumSums ][ N_LANE ];

        const int l_coeff_i = l_coeff_idx[bvr_idx];

        int r_coeff_i[N_LANE];
        for(int n = 0; n < N_LANE; n++) {
            const __fp16* src = r_coeff_cache + r_coeff_i[n] * RNumSums;
            for (int r = 0; r < RNumSums; ++r)
                lane_tile[r][n] = src[r];
        }
        

        for (int l = 0; l < LNumSums; ++l)
        {
            __fp16 a = static_cast<float>(l_coeff_cache[l_coeff_i * LNumSums + l]);
            float16x8_t a_vec = vdupq_n_f16(a);

            for (int r = 0; r < RNumSums; ++r)
            {
                float16x8_t b0 = vld1q_f16(&lane_tile[r][0]);   // lane 0‥7
                float16x8_t b1 = vld1q_f16(&lane_tile[r][8]);   // lane 8‥15

                //  popc[l][r] 16lane → 4×uint32x4_t로 나눠 변환
                uint8x16_t pc_u8 = popc_cache[l][r];
                float16x8_t pc0 = vcvtq_f16_u16( vmovl_u8(vget_low_u8 (pc_u8)) );
                float16x8_t pc1 = vcvtq_f16_u16( vmovl_u8(vget_high_u8(pc_u8)) );

                float16x8_t ab0 = vmulq_f16(a_vec, b0);   // lanes 0‥7
                float16x8_t ab1 = vmulq_f16(a_vec, b1);   // lanes 8‥15

                // (2) FMLALQ: FP16 → FP32 누적
                acc0 = vfmlalq_low_f16 (acc0, ab0, pc0);  // lanes 0‥3
                acc1 = vfmlalq_high_f16(acc1, ab0, pc0);  // lanes 4‥7
                acc2 = vfmlalq_low_f16 (acc2, ab1, pc1);  // lanes 8‥11
                acc3 = vfmlalq_high_f16(acc3, ab1, pc1);  // lanes 12‥15

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

template<
    typename  LIndexT, typename  RIndexT,
    int       LNumSums, int      RNumSums>
void* worker_fn(void* arg_void)
{
    using Arg = WorkerArg<LIndexT,RIndexT,LNumSums,RNumSums>;
    Arg* a = static_cast<Arg*>(arg_void);

    int bvr_per_K = a->K / K_PER_BVR; // Number of BVRs per K

    int N_PACK = a->N / N_LANE; // Number of 16-lane packs

    for (int n_pack = a->n_begin / N_LANE; n_pack < a->n_end / N_LANE; ++n_pack)
    {

        const uint8_t* r_pack =
              a->r_bvr + (n_pack * a->K * RNumSums * N_LANE);

        const RIndexT* r_coeff_idx =
            (const RIndexT*)(a->r_coeff_idx) + n_pack * N_LANE * bvr_per_K;

        __fp16* out_pack = a->out  + n_pack * N_LANE;
        const __fp16* bias_pack = a->bias ? (a->bias + n_pack * N_LANE) : nullptr;

        simd_kernel_1x16<LIndexT,RIndexT,LNumSums,RNumSums>(
            a->l_bvr, (const LIndexT*)(a->l_coeff_idx), a->l_coeff_cache,
            r_pack, r_coeff_idx, a->r_coeff_cache,
            bias_pack, out_pack, a->K);

        if(n_pack == 0 && a->n_begin == 0) {
            // First worker, write the results back to the output tensor
            for (int n = 0; n < N_LANE; ++n) {
                a->out[n] = out_pack[n];
            }
        } 
    }

    return nullptr;
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
    int M, int N, int K, int num_threads)
{

    std::vector<std::thread> threads;
    std::vector<WorkerArg<LIndexT,RIndexT,LNumSums,RNumSums>> args(num_threads);
    

    const int chunk = (N + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t)
    {
        int n0 = t * chunk;
        int n1 = std::min(n0 + chunk, N);

        args[t] = {
            l_bvr, l_coeff_idx, l_coeff_cache,
            r_bvr, r_coeff_idx, r_coeff_cache,
            bias, out,
            K, N, n0, n1
        };

        threads.emplace_back([&, t]() {
            worker_fn<LIndexT, RIndexT, LNumSums, RNumSums>(&args[t]);
        });

    }

    for (auto& th : threads) th.join();
}

template <typename LIndexT, typename RIndexT>
void launch_cpu_sbvr_kernel_wrapper(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx,__fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size, int num_threads)
{
    KernelLaunchFn kernel_list[] = {
        // <LIndexT, RIndexT, LNumSums, RNumSums>
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 4, 4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 4, 8>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 8, 4>,
        sbvr_mm_cpu_1xN<LIndexT, RIndexT, 8, 8>,
    };
    int kernel_idx = (l_num_sums / 4 - 1) * 2 + (r_num_sums / 4 - 1);

    kernel_list[kernel_idx](
        l_bvr, l_coeff_idx, l_coeff_cache,
        r_bvr, r_coeff_idx, r_coeff_cache,
        bias, out,
        M, N, K, num_threads);
        
}

void launch_cpu_sbvr_mm_1xN(
    uint8_t* l_bvr, void* l_coeff_idx, __fp16* l_coeff_cache,
    uint8_t* r_bvr, void* r_coeff_idx,__fp16* r_coeff_cache,
    __fp16* bias, __fp16* out,
    int M, int N, int K,
    int l_num_sums, int r_num_sums,
    int l_cache_size, int r_cache_size, int num_threads)
{
    
    const bool use_l_uint8 = (l_cache_size <= 256);
    const bool use_r_uint8 = (r_cache_size <= 256);

    // Each num_sums must be 4 or 8
    const bool supported_num_sums = 
        (l_num_sums == 4 || l_num_sums == 8) &&
        (r_num_sums == 4 || r_num_sums == 8);

    
    if (supported_num_sums && use_l_uint8 && use_r_uint8) {
        // Use uint8_t for both left and right BVRs
        if(use_l_uint8 && use_r_uint8) {
            launch_cpu_sbvr_kernel_wrapper<uint8_t, uint8_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                l_cache_size, r_cache_size, num_threads);
        } else if (use_l_uint8 && !use_r_uint8) {
            // Use uint8_t for left BVR and uint16_t for right BVR
            launch_cpu_sbvr_kernel_wrapper<uint8_t, uint16_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                l_cache_size, r_cache_size, num_threads);
        } else if (!use_l_uint8 && use_r_uint8) {
            // Use uint16_t for left BVR and uint8_t for right BVR
            launch_cpu_sbvr_kernel_wrapper<uint16_t, uint8_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                l_cache_size, r_cache_size, num_threads);
        } else {
            // Use uint16_t for both left and right BVRs
            launch_cpu_sbvr_kernel_wrapper<uint16_t, uint16_t>(
                l_bvr, l_coeff_idx, l_coeff_cache,
                r_bvr, r_coeff_idx, r_coeff_cache,
                bias, out,
                M, N, K,
                l_num_sums, r_num_sums,
                l_cache_size, r_cache_size, num_threads);
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
    assert (l_bvr.size(0) == r_bvr.size(1));

    auto out = torch::empty({M, N},
                         torch::dtype(torch::kFloat16).device(l_bvr.device()));
    __fp16* bias_ptr = nullptr;
    if (bias.size(0) == N)
        bias_ptr = reinterpret_cast<__fp16*>(bias.data_ptr<at::Half>());


    int num_threads = std::thread::hardware_concurrency();
    
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
        l_cache_size, r_cache_size,
        num_threads);

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
}