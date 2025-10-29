#!/bin/bash
# Simple benchmark downloader for Linux/Mac

echo "============================================"
echo "Chip Placement Benchmark Downloader"
echo "============================================"
echo ""
echo "This will download:"
echo "  - ICCAD04 (IBM): ~150MB"
echo "  - ISPD2005: ~300MB"
echo ""

python3 download_benchmarks.py --all || python download_benchmarks.py --all
