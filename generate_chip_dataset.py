#!/usr/bin/env python
"""
Generate Chip Placement Dataset
Standalone script that works without module import issues
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from DatasetCreator.loadGraphDatasets.ChipDatasetGenerator import ChipDatasetGenerator

def generate_chip_dataset(dataset_name="Chip_small", seed=123, modes=["train", "val", "test"]):
    """
    Generate chip placement dataset for all splits

    Args:
        dataset_name: Name of dataset (Chip_small, Chip_dummy, etc.)
        seed: Random seed
        modes: List of splits to generate (train, val, test)
    """

    for mode in modes:
        print("="*70)
        print(f"GENERATING {mode.upper()} DATASET")
        print("="*70)

        config = {
            "dataset_name": dataset_name,
            "problem": "ChipPlacement",
            "mode": mode,
            "seed": seed,
            "save": True,  # Save to disk
            "parent": False,
            "diff_ps": False,
            "gurobi_solve": False,
            "licence_base_path": "",
            "time_limit": 60.0,
            "thread_fraction": 1.0,
        }

        print(f"Config: {dataset_name}, seed={seed}, mode={mode}")
        print(f"This will generate:")
        if mode == "train":
            print(f"  - ~4000 training instances")
        elif mode == "val":
            print(f"  - ~500 validation instances")
        elif mode == "test":
            print(f"  - ~1000 test instances")
        print()

        # Create generator
        generator = ChipDatasetGenerator(config)

        # Generate dataset
        generator.generate_dataset()

        print(f"\n✓ {mode.upper()} dataset generated successfully!")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Chip Placement Dataset')
    parser.add_argument('--dataset', default='Chip_small',
                       choices=['Chip_small', 'Chip_dummy', 'Chip_large'],
                       help='Dataset name')
    parser.add_argument('--seed', default=123, type=int,
                       help='Random seed (must match training seed!)')
    parser.add_argument('--modes', default=['train', 'val', 'test'], nargs='+',
                       choices=['train', 'val', 'test'],
                       help='Which splits to generate')

    args = parser.parse_args()

    print("="*70)
    print("CHIP PLACEMENT DATASET GENERATION")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Modes: {args.modes}")
    print()
    print("Dataset will have:")
    print("  - Density: 75-90% (ChipDiffusion-style)")
    print("  - Rent's rule: t=128, min_terminals=4")
    print("  - Overlaps: allowed (model learns to fix)")
    print("  - Components: 100-250 per instance")
    print("  - HPWL: 500-2000 (random placement)")
    print()
    input("Press Enter to start generation (or Ctrl+C to cancel)...")
    print()

    # Generate dataset
    generate_chip_dataset(
        dataset_name=args.dataset,
        seed=args.seed,
        modes=args.modes
    )

    print("="*70)
    print("ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*70)
    print()
    print("Dataset location:")
    print(f"  DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{args.dataset}/")
    print()
    print("Next step: Start training with:")
    print(f"  python argparse_ray_main.py --EnergyFunction ChipPlacement --IsingMode {args.dataset} --seed {args.seed} --overlap_weight 5000 --boundary_weight 0.0 ...")
    print()
