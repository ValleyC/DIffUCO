import sys
import jax.numpy as jnp
import flax.linen as nn

# Test if RLHead has the changes
from Networks.Modules.HeadModules.RLHead import RLHeadModule_agg_before
import inspect

print("=" * 60)
print("RLHeadModule_agg_before class inspection:")
print("=" * 60)

# Check class signature
sig = inspect.signature(RLHeadModule_agg_before)
print(f"Parameters: {list(sig.parameters.keys())}")

# Check if continuous_dim exists
annotations = RLHeadModule_agg_before.__annotations__
print(f"Annotations: {annotations}")
print(f"Has continuous_dim: {'continuous_dim' in annotations}")

# Check the setup method
setup_source = inspect.getsource(RLHeadModule_agg_before.setup)
print("\nsetup() method source (first 500 chars):")
print(setup_source[:500])

# Check the __call__ method
call_source = inspect.getsource(RLHeadModule_agg_before.__call__)
print("\n__call__() method source (first 800 chars):")
print(call_source[:800])

# Check if "Values" is in the __call__ method
if '"Values"' in call_source or "'Values'" in call_source:
    print("\n✓ 'Values' key is present in __call__ method")
else:
    print("\n✗ ERROR: 'Values' key NOT found in __call__ method")

print("=" * 60)
