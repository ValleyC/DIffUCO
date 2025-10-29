@echo off
REM Simple benchmark downloader for Windows

echo ============================================
echo Chip Placement Benchmark Downloader
echo ============================================
echo.
echo This will download:
echo   - ICCAD04 (IBM): ~150MB
echo   - ISPD2005: ~300MB
echo.

python download_benchmarks.py --all

pause
