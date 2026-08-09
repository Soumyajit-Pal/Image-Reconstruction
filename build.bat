@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44.35207 >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to initialize MSVC environment.
    exit /b 1
)
echo [OK] MSVC compiler activated.

set PYTHONIOENCODING=utf-8
call C:\Users\palbi\miniconda3\condabin\conda.bat activate denoiser_env
echo [OK] Conda environment activated.

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set DISTUTILS_USE_SDK=1
echo [BUILD] CUDA_HOME=%CUDA_HOME%
echo [BUILD] Compiling CUDA extension...
python setup.py build_ext --inplace
if errorlevel 1 (
    echo [FAIL] build_ext failed.
    exit /b 1
) else (
    echo [OK] CUDA extension installed.
)
