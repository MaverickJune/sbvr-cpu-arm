
#include <cuda_runtime.h>

__global__ void my_add_kernel(float* x, float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] + y[idx];
    }
}

void launch_my_add_kernel(float* x, float* y, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    my_add_kernel<<<blocks, threads>>>(x, y, out, size);
}
