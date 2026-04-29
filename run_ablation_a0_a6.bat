@echo off
REM Run all LQ-ICIF ablation variants A0-A6 sequentially.
REM Use from the activated conda environment:
REM   run_ablation_a0_a6.bat

python run_ablation_a0_a6.py --config config.yml %*
