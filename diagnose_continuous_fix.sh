#!/bin/bash
# Shell script version of the diagnostic
# Checks if continuous_dim fix is in place

echo "======================================================================="
echo "DiffUCO Continuous Architecture Diagnostic (Shell Version)"
echo "======================================================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

echo "Working directory: $SCRIPT_DIR"
echo ""

# Test 1: Check __init__.py signature
echo "TEST 1: Checking get_GNN_model signature..."
if grep -q "def get_GNN_model.*continuous_dim" Networks/Modules/__init__.py; then
    echo "✓ PASS: continuous_dim parameter found"
    grep -n "def get_GNN_model" Networks/Modules/__init__.py | head -1
    test1=0
else
    echo "✗ FAIL: continuous_dim parameter NOT found"
    echo "Current signature:"
    grep -n "def get_GNN_model" Networks/Modules/__init__.py | head -1
    test1=1
fi
echo ""

# Test 2: Check for continuous_dim check
echo "TEST 2: Checking for continuous_dim > 0 check..."
if grep -q "if continuous_dim > 0:" Networks/Modules/__init__.py; then
    echo "✓ PASS: continuous_dim check found"
    grep -n -A 2 "if continuous_dim > 0:" Networks/Modules/__init__.py | head -4
    test2=0
else
    echo "✗ FAIL: continuous_dim check NOT found"
    test2=1
fi
echo ""

# Test 3: Check for print statement
echo "TEST 3: Checking for ContinuousHead print statement..."
if grep -q "Using ContinuousHead" Networks/Modules/__init__.py; then
    echo "✓ PASS: Print statement found"
    grep -n "Using ContinuousHead" Networks/Modules/__init__.py
    test3=0
else
    echo "✗ FAIL: Print statement NOT found"
    test3=1
fi
echo ""

# Test 4: Check DiffModel.py passes continuous_dim
echo "TEST 4: Checking DiffModel passes continuous_dim..."
if grep -q "get_GNN_model.*self.continuous_dim" Networks/DiffModel.py; then
    echo "✓ PASS: DiffModel passes continuous_dim"
    grep -n "get_GNN_model.*self.continuous_dim" Networks/DiffModel.py
    test4=0
else
    echo "✗ FAIL: DiffModel does NOT pass continuous_dim"
    echo "Current call:"
    grep -n "get_GNN_model" Networks/DiffModel.py | grep -v "from"
    test4=1
fi
echo ""

# Test 5: Check conditional head instantiation
echo "TEST 5: Checking conditional head instantiation..."
if grep -q "if self.continuous_dim > 0:" Networks/DiffModel.py; then
    echo "✓ PASS: Conditional head instantiation found"
    grep -n -A 4 "if self.continuous_dim > 0:" Networks/DiffModel.py | head -7
    test5=0
else
    echo "✗ FAIL: Conditional head instantiation NOT found"
    test5=1
fi
echo ""

# Test 6: Check ContinuousHead.py exists
echo "TEST 6: Checking ContinuousHead.py exists..."
if [ -f "Networks/Modules/HeadModules/ContinuousHead.py" ]; then
    echo "✓ PASS: ContinuousHead.py found"
    test6=0
else
    echo "✗ FAIL: ContinuousHead.py NOT found"
    test6=1
fi
echo ""

# Calculate total
total_failures=$((test1 + test2 + test3 + test4 + test5 + test6))

# Summary
echo "======================================================================="
echo "SUMMARY"
echo "======================================================================="
echo ""

if [ $total_failures -eq 0 ]; then
    echo "✓✓✓ ALL TESTS PASSED! ✓✓✓"
    echo ""
    echo "Your code has the continuous_dim fix."
    echo ""
    echo "When you run training, you should see:"
    echo "  'Using ContinuousHead for continuous_dim=2'"
    echo ""
    echo "And model parameters should have:"
    echo "  'HeadModel': {'mean_layer': {...}, 'log_var_layer': {...}}"
    echo ""
    echo "NOT:"
    echo "  'HeadModel': {'probMLP': {...}}"
    echo ""
    echo "======================================================================="
    echo "ACTION: Delete old checkpoints and re-train!"
    echo "======================================================================="
    echo "rm -rf ~/ray_results/Diff-PPO_*"
    exit 0
else
    echo "✗✗✗ $total_failures TESTS FAILED ✗✗✗"
    echo ""
    echo "Your code does NOT have the fix!"
    echo ""
    echo "======================================================================="
    echo "ACTIONS TO FIX:"
    echo "======================================================================="
    echo ""

    if [ $test1 -ne 0 ] || [ $test2 -ne 0 ] || [ $test3 -ne 0 ]; then
        echo "1. Fix Networks/Modules/__init__.py:"
        echo "   git checkout Networks/Modules/__init__.py"
        echo "   git pull"
        echo ""
    fi

    if [ $test4 -ne 0 ] || [ $test5 -ne 0 ]; then
        echo "2. Fix Networks/DiffModel.py:"
        echo "   git checkout Networks/DiffModel.py"
        echo "   git pull"
        echo ""
    fi

    echo "3. Clear Python cache:"
    echo "   find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null"
    echo "   find . -name '*.pyc' -delete"
    echo ""
    echo "4. Re-run this script to verify:"
    echo "   bash diagnose_continuous_fix.sh"
    echo ""
    echo "======================================================================="
    exit 1
fi
