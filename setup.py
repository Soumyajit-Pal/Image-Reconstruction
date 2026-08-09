from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="agg_module",
    ext_modules=[
        CUDAExtension(
            name="agg_module",
            sources=["binding.cpp", "aggregation.cu"],
            extra_compile_args={
                "cxx": ["/Zc:preprocessor"],
                "nvcc": ["-Xcompiler", "/Zc:preprocessor"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

