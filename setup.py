
from setuptools import setup, find_packages
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
                'sbvr/kernels/sbvr_kernel.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
    install_requires=requirements
)
