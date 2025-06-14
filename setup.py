import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# Read Python package dependencies
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Paths
this_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_dir = os.path.join(this_dir, 'cutlass', 'include')
sbvr_include_dir = os.path.join(this_dir, 'sbvr', 'include')

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
            include_dirs=[
                cutlass_dir,
                sbvr_include_dir,
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '--ftz=true',
                    '-Xptxas=-v', 
                ],
            },
        ),
        # CppExtension (for CPU)
        CppExtension(
            name='sbvr.sbvr_cpu',
            sources=[
                'sbvr/kernels/sbvr_kernel.cpp',  # New CPU kernel file
            ],
            include_dirs=[
                sbvr_include_dir,
            ],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-mcpu=native',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=requirements,
)
