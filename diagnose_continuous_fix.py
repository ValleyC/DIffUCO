#!/usr/bin/env python3
"""
Diagnostic script to check if continuous_dim fix is properly installed.
This script verifies that the model will use ContinuousHead for chip placement.

Run this BEFORE training to ensure the architecture is correct.
"""

import sys
import os

print("=" * 70)
print("DiffUCO Continuous Architecture Diagnostic")
print("=" * 70)
print()

# Test 1: Check if we can import get_GNN_model
print("TEST 1: Importing get_GNN_model...")
try:
    from Networks.Modules import get_GNN_model
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check function signature
print("\nTEST 2: Checking function signature...")
import inspect
sig = inspect.signature(get_GNN_model)
params = list(sig.parameters.keys())
print(f"  Parameters: {params}")

if 'continuous_dim' in params:
    print("  ✓ PASS: continuous_dim parameter exists")
    test2_pass = True
else:
    print("  ✗ FAIL: continuous_dim parameter MISSING")
    print("  Your Networks/Modules/__init__.py is NOT updated!")
    test2_pass = False

# Test 3: Check what file Python is actually loading
print("\nTEST 3: Checking file location...")
file_location = inspect.getfile(get_GNN_model)
print(f"  Python is loading from: {file_location}")

expected_path = os.path.join(os.getcwd(), "Networks", "Modules", "__init__.py")
if os.path.samefile(file_location, expected_path):
    print(f"  ✓ Correct file: {expected_path}")
    test3_pass = True
else:
    print(f"  ⚠ WARNING: Expected {expected_path}")
    print(f"  But loading from: {file_location}")
    test3_pass = False

# Test 4: Call get_GNN_model with continuous_dim=2
print("\nTEST 4: Calling get_GNN_model with continuous_dim=2...")
try:
    GNNModel, HeadModel = get_GNN_model("normal", "PPO", continuous_dim=2)
    head_name = HeadModel.__name__
    print(f"  Returned Head: {head_name}")

    if head_name == "ContinuousHead":
        print("  ✓ PASS: Returns ContinuousHead")
        test4_pass = True
    else:
        print(f"  ✗ FAIL: Returns {head_name} instead of ContinuousHead")
        print("  The continuous_dim check in __init__.py is WRONG!")
        test4_pass = False
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    test4_pass = False

# Test 5: Import and check ContinuousHead
print("\nTEST 5: Importing ContinuousHead...")
try:
    from Networks.Modules.HeadModules.ContinuousHead import ContinuousHead
    print("  ✓ ContinuousHead import successful")
    test5_pass = True
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    test5_pass = False

# Test 6: Check DiffModel.py passes continuous_dim
print("\nTEST 6: Checking DiffModel.py...")
diffmodel_path = os.path.join(os.getcwd(), "Networks", "DiffModel.py")
if os.path.exists(diffmodel_path):
    with open(diffmodel_path, 'r') as f:
        content = f.read()

    if 'get_GNN_model(self.EncoderModel, self.train_mode, self.continuous_dim)' in content:
        print("  ✓ PASS: DiffModel passes continuous_dim")
        test6a_pass = True
    else:
        print("  ✗ FAIL: DiffModel does NOT pass continuous_dim")
        print("  Check Networks/DiffModel.py line ~51")
        test6a_pass = False

    if 'if self.continuous_dim > 0:' in content:
        print("  ✓ PASS: DiffModel has conditional head instantiation")
        test6b_pass = True
    else:
        print("  ✗ FAIL: DiffModel missing conditional head instantiation")
        print("  Check Networks/DiffModel.py line ~80")
        test6b_pass = False
else:
    print(f"  ✗ FAIL: DiffModel.py not found at {diffmodel_path}")
    test6a_pass = False
    test6b_pass = False

# Test 7: Check __init__.py has the print statement
print("\nTEST 7: Checking for debug print statement...")
init_path = os.path.join(os.getcwd(), "Networks", "Modules", "__init__.py")
if os.path.exists(init_path):
    with open(init_path, 'r') as f:
        content = f.read()

    if 'Using ContinuousHead' in content:
        print("  ✓ PASS: Print statement found")
        print("  Training log should show: 'Using ContinuousHead for continuous_dim=2'")
        test7_pass = True
    else:
        print("  ✗ FAIL: Print statement NOT found")
        test7_pass = False
else:
    print(f"  ✗ FAIL: __init__.py not found")
    test7_pass = False

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_tests = [test2_pass, test3_pass, test4_pass, test5_pass,
             test6a_pass, test6b_pass, test7_pass]
passed = sum(all_tests)
total = len(all_tests)

print(f"\nTests Passed: {passed}/{total}")

if all(all_tests):
    print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("\nYour code has the continuous_dim fix.")
    print("\nWhen you run training, you should see:")
    print("  'Using ContinuousHead for continuous_dim=2'")
    print("\nAnd model parameters should have:")
    print("  'HeadModel': {'mean_layer': {...}, 'log_var_layer': {...}}")
    print("\nNOT:")
    print("  'HeadModel': {'probMLP': {...}}")
    print("\n" + "=" * 70)
    print("ACTION: You can now train with confidence!")
    print("=" * 70)
else:
    print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("\nYour code does NOT have the fix or is incorrectly configured.")
    print("\n" + "=" * 70)
    print("ACTIONS TO FIX:")
    print("=" * 70)

    if not test2_pass:
        print("\n1. Update Networks/Modules/__init__.py:")
        print("   - Function signature must be: def get_GNN_model(Model_name, train_mode, continuous_dim=0)")

    if not test4_pass:
        print("\n2. Fix continuous_dim check in Networks/Modules/__init__.py:")
        print("   - Add BEFORE the PPO check:")
        print("     if continuous_dim > 0:")
        print("         Head_name = 'ContinuousHead'")
        print("         print(f'Using ContinuousHead for continuous_dim={continuous_dim}')")

    if not test6a_pass:
        print("\n3. Update Networks/DiffModel.py line ~51:")
        print("   - Change to: GNNModel, HeadModel = get_GNN_model(self.EncoderModel, self.train_mode, self.continuous_dim)")

    if not test6b_pass:
        print("\n4. Update Networks/DiffModel.py line ~80:")
        print("   - Add conditional head instantiation:")
        print("     if self.continuous_dim > 0:")
        print("         self.HeadModel = HeadModel(continuous_dim=self.continuous_dim, dtype=dtype)")
        print("     else:")
        print("         self.HeadModel = HeadModel(n_features_list_prob=self.n_features_list_prob, dtype=dtype)")

    if not test3_pass:
        print("\n5. Clear Python cache:")
        print("   find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null")
        print("   find . -name '*.pyc' -delete")

    print("\n" + "=" * 70)
    print("After fixing, re-run this script to verify!")
    print("=" * 70)

print()
sys.exit(0 if all(all_tests) else 1)
