from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='sbvr',
    packages=['sbvr', 'sbvr.kernels'],
    ext_modules=[
        CUDAExtension(
            name='sbvr.sbvr_cuda',
            sources=[
                'sbvr/kernels/sbvr_ops.cpp', 
                'sbvr/kernels/sbvr_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                         '--use_fast_math',
                         '--ftz=true',
                         '-Xptxas="-v"']
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=requirements
)
