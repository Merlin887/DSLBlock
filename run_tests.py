"""
Simplified test runner entry point.
"""

import torch

from infrastructure.test_runner import TestRunner

if __name__ == "__main__":
    # Set deterministic behavior
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Test suites to run
    test_suite_paths = [
        './test_options/comparison_downscaled.yaml'
    ]

    # Run each test suite
    for path in test_suite_paths:
        print(f"\n=== Running: {path} ===")
        runner = TestRunner(path)
        runner.run()
        print(f"=== Completed: {path} ===\n")