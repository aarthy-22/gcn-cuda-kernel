# faced many build errors with cmake so used setuptools instead
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gcn_agg_cuda',
    ext_modules=[
        CUDAExtension('gcn_agg_cuda', [
            'gcn_agg.cu',
            'pybind11_gcn_agg.cpp',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })