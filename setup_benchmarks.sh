#!/bin/bash
# ChipDiffusion Benchmark Setup Script
# This script sets up the environment and runs the benchmark preparation

set -e  # Exit on error

echo "======================================================================"
echo "ChipDiffusion Benchmark Preparation Setup"
echo "======================================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 not found. Please install Python 3 and pip."
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install -r requirements_benchmarks.txt

# Make the Python script executable
chmod +x prepare_chipdiffusion_benchmarks.py

# Run the preparation script
echo ""
echo "Running benchmark preparation..."
python3 prepare_chipdiffusion_benchmarks.py

echo ""
echo "======================================================================"
echo "Setup complete!"
echo "======================================================================"
echo "Datasets are available in: datasets/graph/"
echo "  - datasets/graph/iccad04/     (Full IBM benchmarks)"
echo "  - datasets/graph/ispd2005/    (Full ISPD2005 benchmarks)"
echo "  - datasets/graph/macro-ibm/   (Macro-only IBM)"
echo "  - datasets/graph/macro-ispd/  (Macro-only ISPD2005)"
