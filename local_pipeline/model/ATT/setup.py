import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

include_dirs = torch.utils.cpp_extension.include_paths()
print(include_dirs)
include_dirs.append('/media/yanshi/windows/attention_kernel/src')
include_dirs.append('/media/yanshi/windows/attention_kernel/pybind11-master/include')
print(include_dirs)

setup(
    name="attention_package",
    version="0.2",
    description="attention layer",
    # url="https://github.com/jbarker-nvidia/pytorch-correlation",
    author="Saurus",
    author_email="jia1saurus@gmail.com",
    ext_modules = [
        CUDAExtension(name='at_cuda', 
        include_dirs = include_dirs,
        sources=['src/attention_kernel.cu', 'src/attention_cuda.cpp'])
    ],
    cmdclass={
        'build_ext' : BuildExtension
    }
)
