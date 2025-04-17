
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>

__global__ void naive_sbvr_mm(uint32_t* l_bvr,
                              uint8_t* l_coeff_idx,
                              uint8_t* l_bias_idx,
                              __half* l_coeff_cache,
                              __half* l_bias_cache,
                              uint32_t* r_bvr,
                              uint8_t* r_coeff_idx,
                              uint8_t* r_bias_idx,
                              __half* r_coeff_cache,
                              __half* r_bias_cache,
                              __half* out,
                              int out_rows,
                              int out_cols,
                              int l_num_sums,
                              int r_num_sums,
                              int cgroup_per_inner_vec,
                              int bvr_per_cgroup,
                              int cache_size,
                              int num_cgroups) 
{
    // Tensor shapes:
    // l_bvr: [out_rows, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // l_coeff_idx: [num_cgroups]
    // l_bias_idx: [num_cgroups]
    // l_coeff_cache: [cache_size, num_sums]
    // l_bias_cache: [cache_size]
    // r_bvr: [out_cols, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // r_coeff_idx: [num_cgroups]
    // r_bias_idx: [num_cgroups]
    // r_coeff_cache: [cache_size, num_sums]
    // r_bias_cache: [cache_size]

    __half coeff_mult[13][13];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = out_rows * out_cols;

    for (int i = tid; i < total_outputs; i += blockDim.x * gridDim.x) 
    {
        int row = i / out_cols;
        int col = i % out_cols;

        // Initialize output
        float sum = 0.0f;

        for (int cg_idx = 0; cg_idx < cgroup_per_inner_vec; cg_idx++)
        {
            int l_coeff_cache_idx = 
                l_coeff_idx[row * cgroup_per_inner_vec + cg_idx];
            int l_bias_cache_idx = 
                l_bias_idx[row * cgroup_per_inner_vec + cg_idx];
            __half* l_coeff_ptr = 
                &l_coeff_cache[l_coeff_cache_idx * l_num_sums];
            __half l_bias = l_bias_cache[l_bias_cache_idx];
            int r_coeff_cache_idx = 
                r_coeff_idx[col * cgroup_per_inner_vec + cg_idx];
            int r_bias_cache_idx =
                 r_bias_idx[col * cgroup_per_inner_vec + cg_idx];
            __half* r_coeff_ptr = 
                &r_coeff_cache[r_coeff_cache_idx * r_num_sums];
            __half r_bias = r_bias_cache[r_bias_cache_idx];

            // Precompute the coefficient multiplications
            coeff_mult[0][0] = __hmul(l_bias, r_bias);
            for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
            {
                coeff_mult[0][r_idx + 1] = __hmul(l_bias, r_coeff_ptr[r_idx]);
            }
            for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
            {
                __half l_coeff = l_coeff_ptr[l_idx];
                coeff_mult[l_idx + 1][0] = __hmul(l_coeff, r_bias);
                for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                {
                    coeff_mult[l_idx + 1][r_idx + 1] = 
                        __hmul(l_coeff, r_coeff_ptr[r_idx]);
                }
            }

            for (int bvr_idx = 0; bvr_idx < bvr_per_cgroup; bvr_idx++)
            {
                sum += 32.0 * __half2float(coeff_mult[0][0]);
                for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                {
                    uint32_t r = r_bvr[
                        col * cgroup_per_inner_vec * 
                        bvr_per_cgroup * r_num_sums +
                        r_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                        cg_idx * bvr_per_cgroup + bvr_idx];
                    float r_popc = (float)__popc(r);
                    sum += r_popc * __half2float(coeff_mult[0][r_idx + 1]);
                }
                for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
                {
                    uint32_t l = l_bvr[
                        row * cgroup_per_inner_vec * 
                        bvr_per_cgroup * l_num_sums +
                        l_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                        cg_idx * bvr_per_cgroup + bvr_idx];
                    float l_popc = (float)__popc(l);
                    sum += l_popc * __half2float(coeff_mult[l_idx + 1][0]);
                    for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                    {
                        uint32_t lr = l & r_bvr[
                            col * cgroup_per_inner_vec * 
                            bvr_per_cgroup * r_num_sums +
                            r_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                            cg_idx * bvr_per_cgroup + bvr_idx];
                        float lr_popc = (float)__popc(lr);
                        sum += lr_popc * 
                            __half2float(coeff_mult[l_idx + 1][r_idx + 1]);
                    }
                }
            }

        }

        // Store the result in the output matrix
        out[i] = __float2half(sum);
    }
    
}

__global__ void per_elem_sbvr_mm(uint32_t* l_bvr,
                                 uint8_t* l_coeff_idx,
                                 uint8_t* l_bias_idx,
                                 __half* l_coeff_cache,
                                 __half* l_bias_cache,
                                 uint32_t* r_bvr,
                                 uint8_t* r_coeff_idx,
                                 uint8_t* r_bias_idx,
                                 __half* r_coeff_cache,
                                 __half* r_bias_cache,
                                 __half* out,
                                 int out_rows,
                                 int out_cols,
                                 int l_num_sums,
                                 int r_num_sums,
                                 int cgroup_per_inner_vec,
                                 int bvr_per_cgroup,
                                 int cache_size,
                                 int num_cgroups) 
{
    // Tensor shapes:
    // l_bvr: [out_rows, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // l_coeff_idx: [num_cgroups]
    // l_bias_idx: [num_cgroups]
    // l_coeff_cache: [cache_size, num_sums]
    // l_bias_cache: [cache_size]
    // r_bvr: [out_cols, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // r_coeff_idx: [num_cgroups]
    // r_bias_idx: [num_cgroups]
    // r_coeff_cache: [cache_size, num_sums]
    // r_bias_cache: [cache_size]

    __half coeff_mult[13][13];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = out_rows * out_cols;

    for (int i = tid; i < total_outputs; i += blockDim.x * gridDim.x) 
    {
        int row = i / out_cols;
        int col = i % out_cols;

        // Initialize output
        float sum = 0.0f;

        for (int cg_idx = 0; cg_idx < cgroup_per_inner_vec; cg_idx++)
        {
            int l_coeff_cache_idx = 
                l_coeff_idx[row * cgroup_per_inner_vec + cg_idx];
            int l_bias_cache_idx = 
                l_bias_idx[row * cgroup_per_inner_vec + cg_idx];
            __half* l_coeff_ptr = 
                &l_coeff_cache[l_coeff_cache_idx * l_num_sums];
            __half l_bias = l_bias_cache[l_bias_cache_idx];
            int r_coeff_cache_idx = 
                r_coeff_idx[col * cgroup_per_inner_vec + cg_idx];
            int r_bias_cache_idx =
                 r_bias_idx[col * cgroup_per_inner_vec + cg_idx];
            __half* r_coeff_ptr = 
                &r_coeff_cache[r_coeff_cache_idx * r_num_sums];
            __half r_bias = r_bias_cache[r_bias_cache_idx];

            // Precompute the coefficient multiplications
            coeff_mult[0][0] = __hmul(l_bias, r_bias);
            for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
            {
                coeff_mult[0][r_idx + 1] = __hmul(l_bias, r_coeff_ptr[r_idx]);
            }
            for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
            {
                __half l_coeff = l_coeff_ptr[l_idx];
                coeff_mult[l_idx + 1][0] = __hmul(l_coeff, r_bias);
                for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                {
                    coeff_mult[l_idx + 1][r_idx + 1] = 
                        __hmul(l_coeff, r_coeff_ptr[r_idx]);
                }
            }

            for (int bvr_idx = 0; bvr_idx < bvr_per_cgroup; bvr_idx++)
            {
                sum += 32.0 * __half2float(coeff_mult[0][0]);
                for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                {
                    uint32_t r = r_bvr[
                        col * cgroup_per_inner_vec * 
                        bvr_per_cgroup * r_num_sums +
                        r_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                        cg_idx * bvr_per_cgroup + bvr_idx];
                    float r_popc = (float)__popc(r);
                    sum += r_popc * __half2float(coeff_mult[0][r_idx + 1]);
                }
                for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
                {
                    uint32_t l = l_bvr[
                        row * cgroup_per_inner_vec * 
                        bvr_per_cgroup * l_num_sums +
                        l_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                        cg_idx * bvr_per_cgroup + bvr_idx];
                    float l_popc = (float)__popc(l);
                    sum += l_popc * __half2float(coeff_mult[l_idx + 1][0]);
                    for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                    {
                        uint32_t lr = l & r_bvr[
                            col * cgroup_per_inner_vec * 
                            bvr_per_cgroup * r_num_sums +
                            r_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                            cg_idx * bvr_per_cgroup + bvr_idx];
                        float lr_popc = (float)__popc(lr);
                        sum += lr_popc * 
                            __half2float(coeff_mult[l_idx + 1][r_idx + 1]);
                    }
                }
            }

        }

        // Store the result in the output matrix
        out[i] = __float2half(sum);
    }
    
}

extern "C" void launch_sbvr_mm(
                    uint32_t* l_bvr,
                    uint8_t* l_coeff_idx,
                    uint8_t* l_bias_idx,
                    __half* l_coeff_cache,
                    __half* l_bias_cache,
                    uint32_t* r_bvr,
                    uint8_t* r_coeff_idx,
                    uint8_t* r_bias_idx,
                    __half* r_coeff_cache,
                    __half* r_bias_cache,
                    __half* out,
                    int out_rows,
                    int out_cols,
                    int l_num_sums,
                    int r_num_sums,
                    int cgroup_per_inner_vec,
                    int bvr_per_cgroup,
                    int cache_size,
                    int num_cgroups) 
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount * 8;
    dim3 threads = 32;
    std::cout << "Blocks: " << blocks << ", Threads: " << threads.x << std::endl;
    naive_sbvr_mm<<<blocks, threads>>>(l_bvr,
                                       l_coeff_idx,
                                       l_bias_idx,
                                       l_coeff_cache,
                                       l_bias_cache,
                                       r_bvr,
                                       r_coeff_idx,
                                       r_bias_idx,
                                       r_coeff_cache,
                                       r_bias_cache,
                                       out,
                                       out_rows,
                                       out_cols,
                                       l_num_sums,
                                       r_num_sums,
                                       cgroup_per_inner_vec,
                                       bvr_per_cgroup,
                                       cache_size,
                                       num_cgroups);
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}
