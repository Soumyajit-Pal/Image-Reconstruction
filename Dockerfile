FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and build essentials
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy project files
COPY --chown=user:user . /home/user/app

# Install PyTorch with CUDA 11.8 support and all dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    gradio \
    scikit-image \
    numpy \
    matplotlib \
    Pillow

# Compile the custom C++/CUDA kernels
# The setup.py uses MSVC flags (/Zc:preprocessor) which are Windows-only.
# We create a Linux-compatible setup for the Docker build.
RUN python3 -c "\
from setuptools import setup; \
from torch.utils.cpp_extension import BuildExtension, CUDAExtension; \
setup( \
    name='agg_module', \
    ext_modules=[ \
        CUDAExtension( \
            name='agg_module', \
            sources=['csrc/binding.cpp', 'csrc/aggregation.cu'], \
        ), \
    ], \
    cmdclass={'build_ext': BuildExtension}, \
)" build_ext --inplace

# Switch to non-root user
USER user

EXPOSE 7860

CMD ["python3", "app.py"]
