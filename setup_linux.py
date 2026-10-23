from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="agg_module",
    ext_modules=[
        CUDAExtension(
            name="agg_module",
            sources=["csrc/binding.cpp", "csrc/aggregation.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
