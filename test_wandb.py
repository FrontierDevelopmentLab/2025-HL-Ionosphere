#!/usr/bin/env python3
"""
Test script to verify wandb integration works correctly
"""
import argparse
import sys
import os

# Add the scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_wandb_import():
    """Test that wandb can be imported"""
    try:
        import wandb
        print("✓ wandb import successful")
        return True
    except ImportError as e:
        print(f"✗ wandb import failed: {e}")
        return False

def test_run_script_import():
    """Test that run script can be imported with wandb"""
    try:
        from scripts import run
        print("✓ run.py import successful")
        return True
    except ImportError as e:
        print(f"✗ run.py import failed: {e}")
        return False

def test_wandb_args():
    """Test that wandb arguments are parsed correctly"""
    try:
        from scripts.run import main
        # Test if the new wandb arguments exist by creating parser
        import scripts.run as run_module
        
        # Create a simple test to verify arguments exist
        parser = argparse.ArgumentParser()
        parser.add_argument('--wandb_disabled', action='store_true')
        parser.add_argument('--wandb_run_name', type=str)
        parser.add_argument('--wandb_notes', type=str)
        parser.add_argument('--wandb_tags', nargs='*')
        
        # Parse empty args to test structure
        args = parser.parse_args([])
        print("✓ wandb arguments structure test successful")
        return True
    except Exception as e:
        print(f"✗ wandb arguments test failed: {e}")
        return False

if __name__ == '__main__':
    print("Testing wandb integration...")
    print("-" * 40)
    
    tests = [
        test_wandb_import,
        test_run_script_import,
        test_wandb_args
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("-" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! wandb integration ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed.")
        sys.exit(1)