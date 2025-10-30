"""
Debug script to identify the shape mismatch issue
"""
import pickle
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "Checkpoints/rubq72w5/best_rubq72w5.pickle"

print("=" * 70)
print("SHAPE MISMATCH DEBUGGER")
print("=" * 70)

# Load checkpoint
with open(checkpoint_path, 'rb') as f:
    checkpoint = pickle.load(f)

config = checkpoint['config']
params = checkpoint['params']['params']

print("\n=== CHECKPOINT CONFIG ===")
print(f"n_diffusion_steps: {config['n_diffusion_steps']}")
print(f"n_random_node_features: {config['n_random_node_features']}")
print(f"time_encoding: {config['time_encoding']}")
print(f"continuous_dim: {config.get('continuous_dim', 0)}")
print(f"embedding_dim: {config.get('embedding_dim', 32)}")

print("\n=== ACTUAL PARAMETER SHAPES ===")
# Check node_encoder
node_enc_kernel = params['encode_process_decode']['node_encoder']['mlp']['layers_0']['kernel']
print(f"node_encoder input dim: {node_enc_kernel.shape[2]}")  # (1, input_dim, 256)

# Check W_message (multiple process blocks)
w_msg_kernel = params['encode_process_decode']['process_block_0']['W_message']['kernel']
print(f"W_message input dim: {w_msg_kernel.shape[1]}")  # (1, input_dim, 256)

print("\n=== CALCULATED DIMENSIONS ===")
continuous_dim = config.get('continuous_dim', 0)
n_random_node_features = config['n_random_node_features']
n_diffusion_steps = config['n_diffusion_steps']
time_encoding = config['time_encoding']

if time_encoding == 'one_hot':
    time_dim = n_diffusion_steps
else:
    time_dim = config.get('embedding_dim', 32)

# node_encoder input: continuous_dim + time_dim + n_random_node_features
node_encoder_input = continuous_dim + time_dim + n_random_node_features
print(f"Expected node_encoder input: {continuous_dim} + {time_dim} + {n_random_node_features} = {node_encoder_input}")
print(f"Actual node_encoder input: {node_enc_kernel.shape[2]}")
print(f"Match: {node_encoder_input == node_enc_kernel.shape[2]}")

# W_message input: This is trickier - it depends on the GNN architecture
# Typically it's: node_embedding_dim + neighbor_embedding_dim + edge_features
# Or: concat of various features
print(f"\nW_message input (from checkpoint): {w_msg_kernel.shape[1]}")
print(f"This should be: node_embedding_dim (256) + additional_features (?)")

# Check if there's a pattern
for i in range(16):  # n_message_passes
    block_name = f'process_block_{i}'
    if block_name in params['encode_process_decode']:
        w_msg = params['encode_process_decode'][block_name]['W_message']['kernel']
        print(f"  {block_name}: W_message shape = {w_msg.shape}")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print(f"The checkpoint expects W_message input dim: {w_msg_kernel.shape[1]}")
print(f"During evaluation, it's trying to create: 260")
print(f"Difference: {260 - w_msg_kernel.shape[1]}")
print("\nPossible causes:")
print("1. n_random_node_features mismatch (checkpoint has 2, code default is 5)")
print("2. Some config value not being passed to model initialization")
print("3. The model architecture computation is different during eval vs train")
