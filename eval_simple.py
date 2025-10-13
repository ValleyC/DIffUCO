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
    trainer = TrainMeanField(config)

    # Load parameters
    trainer.params = checkpoint['params']
    print("Loaded parameters\n")

    # Run test evaluation
    print("Running evaluation on test set...")
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
