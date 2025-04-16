
#include <torch/extension.h>

// Declare the kernel
void launch_my_add_kernel(float* x, float* y, float* out, int size);

// PyTorch wrapper
torch::Tensor my_add(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must be the same size");

    auto out = torch::empty_like(x);
    int size = x.numel();

    launch_my_add_kernel(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sbvr_mat_vec_mult", &my_add, "SBVR CUDA GEMV kernel");
}
