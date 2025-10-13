"""
Manual evaluation by loading data directly and running model inference

Usage:
    python eval_manual.py --checkpoint Checkpoints/90r9zq60/best_90r9zq60.pickle --dataset Chip_small --n_samples 5
"""

import sys
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(".")

from train import TrainMeanField

def load_test_data_direct(dataset_name):
    """Load test data directly from pickle"""
    path = f"DatasetCreator/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/test_ChipPlacement_seed_123_solutions.pickle"
    print(f"Loading: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_hpwl_numpy(positions, graph):
    """Compute HPWL using numpy"""
    total = 0.0
    for s, r in zip(graph.senders, graph.receivers):
        if positions.shape[1] == 2:  # 2D positions
            x1, y1 = positions[s]
            x2, y2 = positions[r]
            total += abs(x1 - x2) + abs(y1 - y2)
        else:  # 1D positions (wrong!)
            total += abs(positions[s, 0] - positions[r, 0])
    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='Chip_small')
    parser.add_argument('--n_samples', type=int, default=5)
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        checkpoint = pickle.load(f)

    config = checkpoint['config']

    print("\n" + "="*80)
    print("CONFIG")
    print("="*80)
    print(f"Problem: {config['problem_name']}")
    print(f"Continuous dim: {config.get('continuous_dim', 'NOT SET!')}")
    print(f"n_bernoulli_features: {config.get('n_bernoulli_features', 'NOT SET')}")
    print(f"n_diffusion_steps: {config['n_diffusion_steps']}")
    print("="*80 + "\n")

    # Load test data directly
    test_data = load_test_data_direct(args.dataset)
    print(f"Loaded {len(test_data['H_graphs'])} test instances\n")

    # Initialize trainer to get model
    print("Initializing model...")
    config['wandb'] = False
    config['dataset_name'] = args.dataset

    trainer = TrainMeanField(config)
    trainer.params = checkpoint['params']

    # Unwrap params from pmap
    params_single = jax.tree_util.tree_map(
        lambda x: x[0] if (isinstance(x, jnp.ndarray) and len(x.shape) > 0 and x.shape[0] == 1) else x,
        trainer.params
    )

    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80 + "\n")

    initial_hpwls = []
    generated_hpwls = []

    for i in range(min(args.n_samples, len(test_data['H_graphs']))):
        graph = test_data['H_graphs'][i]
        initial_positions = test_data['positions'][i]

        initial_hpwl = compute_hpwl_numpy(initial_positions, graph)
        initial_hpwls.append(initial_hpwl)

        print(f"Instance {i}:")
        print(f"  Components: {graph.nodes.shape[0]}")
        print(f"  Initial HPWL: {initial_hpwl:.2f}")

        # Run model inference
        try:
            graph_dict = {"graphs": [graph]}
            key = jax.random.PRNGKey(i)

            loss, (log_dict, _) = trainer.TrainerClass.sample(
                params_single,
                graph_dict,
                graph,
                trainer.T,
                key
            )

            # Extract generated positions
            if "metrics" in log_dict and "X_0" in log_dict["metrics"]:
                X_0 = log_dict["metrics"]["X_0"]
            elif "X_0" in log_dict:
                X_0 = log_dict["X_0"]
            else:
                print(f"  ERROR: Cannot find X_0. Keys: {log_dict.keys()}")
                generated_hpwls.append(initial_hpwl)
                continue

            print(f"  X_0 shape: {X_0.shape}")

            # Handle shape [n_nodes, n_basis_states, dim]
            if len(X_0.shape) == 3:
                generated_positions = np.array(X_0[:, 0, :])  # Take first basis state
            elif len(X_0.shape) == 2:
                generated_positions = np.array(X_0)
            else:
                print(f"  ERROR: Unexpected X_0 shape: {X_0.shape}")
                generated_hpwls.append(initial_hpwl)
                continue

            print(f"  Generated positions shape: {generated_positions.shape}")

            if generated_positions.shape[1] != 2:
                print(f"  ERROR: Generated positions should be 2D but got shape {generated_positions.shape}")
                print(f"  This means the model is NOT outputting 2D continuous positions!")
                print(f"  The model was likely NOT trained with continuous_dim=2")
                generated_hpwl = initial_hpwl
            else:
                generated_hpwl = compute_hpwl_numpy(generated_positions, graph)

            generated_hpwls.append(generated_hpwl)
            improvement = (initial_hpwl - generated_hpwl) / initial_hpwl * 100

            print(f"  Generated HPWL: {generated_hpwl:.2f}")
            print(f"  Improvement: {improvement:.1f}%\n")

        except Exception as e:
            print(f"  ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
            generated_hpwls.append(initial_hpwl)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nInitial HPWL (random placement):")
    print(f"  Mean: {np.mean(initial_hpwls):.2f}")
    print(f"  Min:  {np.min(initial_hpwls):.2f}")
    print(f"  Max:  {np.max(initial_hpwls):.2f}")

    if len(generated_hpwls) > 0:
        print(f"\nGenerated HPWL (trained model):")
        print(f"  Mean: {np.mean(generated_hpwls):.2f}")
        print(f"  Min:  {np.min(generated_hpwls):.2f}")
        print(f"  Max:  {np.max(generated_hpwls):.2f}")

        avg_improvement = np.mean([(i - g)/i * 100 for i, g in zip(initial_hpwls, generated_hpwls)])
        print(f"\nAverage Improvement: {avg_improvement:.1f}%")
    else:
        print("\nNo successful inference runs!")

    print("="*80)

if __name__ == '__main__':
    main()
