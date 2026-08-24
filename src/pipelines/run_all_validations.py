"""
Run all production validation checks.

Run with:
    python -m src.pipelines.run_all_validations
"""

import subprocess
from pathlib import Path


def main():
    BASE_DIR = Path(__file__).resolve().parents[2]

    tests = [
        ["python", "-m", "src.pipelines.validate_production"],
        ["python", "-m", "src.features.test_price_perturbation"],
        ["python", "-m", "src.features.test_grouped_rolling"],
    ]

    for cmd in tests:
        print(f"\n=== Running: {' '.join(cmd)} ===")
        result = subprocess.run(cmd, cwd=BASE_DIR)
        if result.returncode != 0:
            raise RuntimeError(f"Test failed: {' '.join(cmd)}")

    print("\n=== ALL VALIDATION TESTS PASSED ===")


if __name__ == "__main__":
    main()