
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdint>

__global__ void naive_cuda_sbvr_mm_T(
    uint32_t* l_bvr,
    uint8_t* l_coeff_idx,
    __half* l_coeff_cache,
    uint32_t* r_bvr,
    uint8_t* r_coeff_idx,
    __half* r_coeff_cache,
    __half* out,
    int out_rows,
    int out_cols,
    int l_num_sums,
    int r_num_sums,
    int l_cache_size,
    int r_cache_size,
    int cgroup_per_inner_vec,
    int bvr_per_cgroup) 
{
    // Tensor shapes:
    // l_bvr: [out_rows, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // l_coeff_idx: [num_cgroups]
    // l_coeff_cache: [cache_size, num_sums]
    // r_bvr: [out_cols, self.num_sums, cgroup_per_inner_vec, bvr_per_cgroup]
    // r_coeff_idx: [num_cgroups]
    // r_coeff_cache: [cache_size, num_sums]

    float coeff_mult[10][10];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = out_rows * out_cols;

    for (int i = tid; i < total_outputs; i += blockDim.x * gridDim.x) 
    {
        int row = i / out_cols;
        int col = i % out_cols;

        // Initialize output
        float sum = 0.0f;

        // if (tid == 0)
        //     printf("row: %d, col: %d, out_rows: %d, out_cols: %d, "
        //         "l_num_sums: %d, r_num_sums: %d, l_cache_size: %d, "
        //         "r_cache_size: %d, cgroup_per_inner_vec: %d, "
        //         "bvr_per_cgroup: %d\n",
        //         row, col, out_rows, out_cols, l_num_sums, r_num_sums,
        //         l_cache_size, r_cache_size, cgroup_per_inner_vec,
        //         bvr_per_cgroup);

        for (int cg_idx = 0; cg_idx < cgroup_per_inner_vec; cg_idx++)
        {
            int l_coeff_cache_idx = 
                l_coeff_idx[(row * cgroup_per_inner_vec + cg_idx) <<
                                (l_cache_size > 256)];
            __half* l_coeff_ptr = 
                &l_coeff_cache[l_coeff_cache_idx * l_num_sums];
            int r_coeff_cache_idx = 
                r_coeff_idx[(col * cgroup_per_inner_vec + cg_idx) <<
                                (r_cache_size > 256)];
            __half* r_coeff_ptr = 
                &r_coeff_cache[r_coeff_cache_idx * r_num_sums];

            // Precompute the coefficient multiplications
            for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
            {
                float l_coeff = __half2float(l_coeff_ptr[l_idx]);
                for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                {
                    coeff_mult[l_idx][r_idx] = 
                        l_coeff * __half2float(r_coeff_ptr[r_idx]);
                    // if (tid == 0)
                    //     printf("cg_idx: %f, l_coeff: %f, "
                    //         "r_coeff: %f, coeff_mult: %f\n", 
                    //         cg_idx, l_coeff, 
                    //         __half2float(r_coeff_ptr[r_idx]), 
                    //         l_coeff * __half2float(r_coeff_ptr[r_idx]));
                }
            }

            for (int bvr_idx = 0; bvr_idx < bvr_per_cgroup; bvr_idx++)
            {
                for (int l_idx = 0; l_idx < l_num_sums; l_idx++)
                {
                    uint32_t l = l_bvr[
                        row * cgroup_per_inner_vec * 
                        bvr_per_cgroup * l_num_sums +
                        l_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                        cg_idx * bvr_per_cgroup + bvr_idx];
                    for (int r_idx = 0; r_idx < r_num_sums; r_idx++)
                    {
                        uint32_t r = r_bvr[
                            col * cgroup_per_inner_vec * 
                            bvr_per_cgroup * r_num_sums +
                            r_idx * cgroup_per_inner_vec * bvr_per_cgroup + 
                            cg_idx * bvr_per_cgroup + bvr_idx];
                        uint32_t lr = l & r;
                        float lr_popc = (float)__popc(lr);
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

__global__ void per_elem_cuda_sbvr_mm_T(
    uint32_t* l_bvr,
    uint8_t* l_coeff_idx,
    __half* l_coeff_cache,
    uint32_t* r_bvr,
    uint8_t* r_coeff_idx,
    __half* r_coeff_cache,
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

    
}

extern "C" void launch_cuda_sbvr_mm_T(
                    uint32_t* l_bvr,
                    uint8_t* l_coeff_idx,
                    __half* l_coeff_cache,
                    uint32_t* r_bvr,
                    uint8_t* r_coeff_idx,
                    __half* r_coeff_cache,
                    __half* out,
                    int out_rows,
                    int out_cols,
                    int l_num_sums,
                    int r_num_sums,
                    int l_cache_size,
                    int r_cache_size,
                    int cgroup_per_inner_vec,
                    int bvr_per_cgroup) 
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount * 8;
    dim3 threads = 32;
    naive_cuda_sbvr_mm_T<<<blocks, threads>>>(
        l_bvr,
        l_coeff_idx,
        l_coeff_cache,
        r_bvr,
        r_coeff_idx,
        r_coeff_cache,
        out,
        out_rows,
        out_cols,
        l_num_sums,
        r_num_sums,
        l_cache_size,
        r_cache_size,
        cgroup_per_inner_vec,
        bvr_per_cgroup);
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}
