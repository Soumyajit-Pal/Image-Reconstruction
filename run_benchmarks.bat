@echo off
call build.bat
echo.
echo === RUNNING DEFAULT BENCHMARK ===
python benchmark.py
echo.
echo === RUNNING LARGER BENCHMARK ===
python benchmark.py --hidden_channels 32 --num_layers 4 --radius 5 --batch_size 4
