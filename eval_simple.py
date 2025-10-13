"""
Simplest possible evaluation - just run the trainer's eval on test set

Usage:
    python eval_simple.py --checkpoint Checkpoints/90r9zq60/best_90r9zq60.pickle --dataset Chip_small
"""

import sys
import argparse
import pickle

sys.path.append(".")

from train import TrainMeanField

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='Chip_small')
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        checkpoint = pickle.load(f)

    # Get config and update dataset
    config = checkpoint['config']
    config['dataset_name'] = args.dataset
    config['wandb'] = False

    print("\n" + "="*80)
    print("CONFIG")
    print("="*80)
    print(f"Dataset: {config['dataset_name']}")
    print(f"Problem: {config['problem_name']}")
    print(f"Continuous dim: {config.get('continuous_dim', 'NOT SET!')}")
    print(f"n_diffusion_steps: {config['n_diffusion_steps']}")
    print(f"n_basis_states: {config['N_basis_states']}")
    print("="*80 + "\n")

    # Initialize trainer
    print("Initializing trainer...")

    # WORKAROUND: Temporarily patch the dataloader statistics computation
    # to skip test set (which causes multiprocessing issues)
    import Data.LoadGraphDataset as LoadGraphDataset
    original_compute_stats = LoadGraphDataset.SolutionDatasetLoader._compute_dataset_statistics

    def patched_compute_stats(self, mode="train"):
        if mode == "test":
            print(f"  Skipping statistics computation for test set (workaround)")
            # Set dummy values
            if self.dataloader_test is not None:
                self.dataloader_test.smallest_n_edges_input_graph = 0
                self.dataloader_test.largest_n_edges_input_graph = 1000
                self.dataloader_test.smallest_n_edges_energy_graph = 0
                self.dataloader_test.largest_n_edges_energy_graph = 1000
                self.dataloader_test.smallest_n_nodes_input_graph = 0
                self.dataloader_test.largest_n_nodes_input_graph = 100
                self.dataloader_test.smallest_n_nodes_energy_graph = 0
                self.dataloader_test.largest_n_nodes_energy_graph = 100
            return
        else:
            return original_compute_stats(self, mode)

    LoadGraphDataset.SolutionDatasetLoader._compute_dataset_statistics = patched_compute_stats

    try:
        trainer = TrainMeanField(config)
    except Exception as e:
        print(f"ERROR during trainer initialization: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # Restore original function
        LoadGraphDataset.SolutionDatasetLoader._compute_dataset_statistics = original_compute_stats

    # Load parameters
    trainer.params = checkpoint['params']
    print("Loaded parameters\n")

    # Check if dataloaders exist
    print("Checking dataloaders...")
    print(f"  dataloader_train: {trainer.dataloader_train is not None}")
    print(f"  dataloader_val: {trainer.dataloader_val is not None}")
    print(f"  dataloader_test: {trainer.dataloader_test is not None}")

    if trainer.dataloader_test is None:
        print("\nERROR: Test dataloader is None!")
        print(f"Dataset path should be:")
        print(f"  DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{config['dataset_name']}/test_ChipPlacement_seed_123_solutions.pickle")

        import os
        test_path = f"DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{config['dataset_name']}/test_ChipPlacement_seed_123_solutions.pickle"
        print(f"\nChecking if file exists: {os.path.exists(test_path)}")

        # Check what files do exist
        dir_path = f"DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{config['dataset_name']}/"
        if os.path.exists(dir_path):
            print(f"\nFiles in {dir_path}:")
            for f in os.listdir(dir_path):
                print(f"  {f}")
        else:
            print(f"\nDirectory does not exist: {dir_path}")

        return

    # Run test evaluation
    print("\nRunning evaluation on test set...")
    eval_dict = trainer.test(mode="test")

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for key, value in eval_dict.items():
        if not key.endswith('_mat'):  # Skip large matrices
            print(f"{key}: {value}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
