from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='dtet',
    ext_modules=[
        CUDAExtension(
            name='dtet',
            sources=['src/main.cpp', 
                    'src/cuda_kernels.cu'],
            include_dirs=['include'],
            extra_compile_args={'cxx': ['-O2'],
                              'nvcc': ['-O2']},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
