#!/usr/bin/env python3
"""
Simple Benchmark Downloader
Downloads ICCAD04 and ISPD2005 benchmarks from official sources
"""

import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path

# Benchmark URLs
BENCHMARKS = {
    'iccad04': {
        'name': 'ICCAD04 (IBM)',
        'url': 'http://vlsicad.eecs.umich.edu/BK/ICCAD04bench/ibmMSWpinsICCAD04Bench_BOOKSHELF.tar.gz',
        'file': 'ibmMSWpinsICCAD04Bench_BOOKSHELF.tar.gz',
        'description': '18 IBM circuits (ibm01-ibm18)'
    },
    'ispd2005': {
        'name': 'ISPD2005',
        'url': 'http://vlsicad.eecs.umich.edu/BK/ISPD05bench/ISPD05_Processed.tar.gz',
        'file': 'ISPD05_Processed.tar.gz',
        'description': '8 circuits (adaptec1-4, bigblue1-4)'
    }
}

def download_file(url, output_path):
    """Download file with progress bar"""
    print(f"\nDownloading: {url}")
    print(f"Saving to: {output_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '=' * filled + '-' * (bar_length - filled)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f'\r  [{bar}] {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='', flush=True)

    try:
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print("\n  Download complete!")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False

def extract_archive(archive_path, extract_dir):
    """Extract tar.gz or tar archive"""
    print(f"\nExtracting: {archive_path.name}")

    try:
        extract_dir.mkdir(parents=True, exist_ok=True)

        if archive_path.name.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif archive_path.name.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(extract_dir)
        elif archive_path.name.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            print(f"  Unknown archive format: {archive_path.suffix}")
            return False

        print(f"  Extracted to: {extract_dir}")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False

def download_benchmark(benchmark_name, base_dir='./benchmarks'):
    """Download and extract a benchmark"""
    if benchmark_name not in BENCHMARKS:
        print(f"Unknown benchmark: {benchmark_name}")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        return False

    info = BENCHMARKS[benchmark_name]

    print(f"\n{'='*70}")
    print(f"Downloading: {info['name']}")
    print(f"{'='*70}")
    print(f"Description: {info['description']}")

    # Create directories
    base_path = Path(base_dir)
    benchmark_dir = base_path / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Download
    archive_path = benchmark_dir / info['file']

    if archive_path.exists():
        print(f"\nArchive already exists: {archive_path}")
        response = input("Re-download? (y/n): ").lower().strip()
        if response == 'y':
            archive_path.unlink()
        else:
            print("Using existing file.")

    if not archive_path.exists():
        if not download_file(info['url'], archive_path):
            return False

    # Extract
    extract_dir = benchmark_dir / 'extracted'
    if not extract_archive(archive_path, extract_dir):
        return False

    print(f"\n{'='*70}")
    print(f"SUCCESS: {info['name']} downloaded!")
    print(f"{'='*70}")
    print(f"Location: {benchmark_dir}")
    print(f"Extracted files: {extract_dir}")

    return True

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download chip placement benchmarks')
    parser.add_argument('--benchmark', type=str, choices=['iccad04', 'ispd2005', 'all'],
                       default='all', help='Which benchmark to download (default: all)')
    parser.add_argument('--base-dir', type=str, default='./benchmarks',
                       help='Base directory for benchmarks (default: ./benchmarks)')

    args = parser.parse_args()

    # Determine which benchmarks to download
    if args.benchmark == 'all':
        benchmarks = list(BENCHMARKS.keys())
    else:
        benchmarks = [args.benchmark]

    print("\n" + "="*70)
    print("Chip Placement Benchmark Downloader")
    print("="*70)
    print(f"\nWill download: {', '.join([BENCHMARKS[b]['name'] for b in benchmarks])}")
    print(f"Destination: {args.base_dir}")

    response = input("\nContinue? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return

    # Download each benchmark
    success_count = 0
    for benchmark in benchmarks:
        if download_benchmark(benchmark, args.base_dir):
            success_count += 1

    print(f"\n{'='*70}")
    print(f"Download Summary")
    print(f"{'='*70}")
    print(f"Successfully downloaded: {success_count}/{len(benchmarks)}")
    print(f"\nBenchmarks are in: {args.base_dir}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
