#!/usr/bin/env python3
"""
Test runner script for AI Research Lab project.
This script runs all unit tests in the project.
"""
import os
import sys
import unittest
import pytest

def run_tests():
    """Run all unit tests in the project"""
    # Print information about the test execution
    print("=" * 80)
    print("Running AI Research Lab unit tests")
    print("=" * 80)
    
    # Add the project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    sys.path.insert(0, project_root)
    
    # Run tests using pytest (handles both standard unittest tests and pytest.mark.asyncio)
    print("\nRunning tests with pytest (including async tests):")
    test_dir = os.path.dirname(__file__)
    pytest_exit_code = pytest.main(["-xvs", test_dir])
    
    if pytest_exit_code == 0:
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(run_tests())